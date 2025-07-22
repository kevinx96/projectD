import cv2
import mediapipe as mp
from ultralytics import YOLO
import argparse
import numpy as np

# --- 初始化所有模型 ---
# 加载您自己训练好的YOLOv8模型
yolo_model = YOLO('D:/project enshu/runs/detect/train2/weights/best.pt') 

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# --- 辅助函数 ---
def calculate_angle(a, b, c):
    """计算由三个点构成的角度 (b为顶点)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def is_pose_in_box(pose_landmarks, box, image_shape):
    """检查一个姿态是否在边界框内"""
    h, w = image_shape
    hip_x = (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x + 
             pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
    hip_y = (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + 
             pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    
    pose_x_px = hip_x * w
    pose_y_px = hip_y * h
    
    x1, y1, x2, y2 = box
    if x1 < pose_x_px < x2 and y1 < pose_y_px < y2:
        return True
    return False

# --- 评分函数定义 (以滑梯为例) ---
def calculate_slide_score(pose_landmarks):
    """根据姿态关键点计算滑梯玩法的分数"""
    score = 100
    
    landmarks = pose_landmarks.landmark
    # 获取所有需要的关键点
    shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_l = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_r = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ankle_r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    # --- 检查所有关键点的可见度(置信度) ---
    required_landmarks = [shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r, nose]
    if not all(p.visibility > 0.6 for p in required_landmarks):
        return "不确定 (关键点被遮挡)"

    # --- 新增规则1: 坐姿是否正确 (髋关节高于膝盖，且膝盖弯曲) ---
    avg_hip_y = (hip_l.y + hip_r.y) / 2
    avg_knee_y = (knee_l.y + knee_r.y) / 2
    avg_knee_angle = (calculate_angle(hip_l, knee_l, ankle_l) + calculate_angle(hip_r, knee_r, ankle_r)) / 2
    
    # 在图像坐标系中, y值越小位置越高
    if avg_hip_y >= avg_knee_y:
        score -= 20
        print(f"扣分: 坐姿不正确 (髋关节低于或等于膝盖)")
    
    if avg_knee_angle > 160: # 膝盖几乎伸直
        score -= 15
        print(f"扣分: 坐姿不正确 (膝盖过于伸直, 角度: {avg_knee_angle:.2f})")

    # --- 新增规则2: 是否头朝下滑行 (脚的Y坐标小于头的Y坐标) ---
    avg_ankle_y = (ankle_l.y + ankle_r.y) / 2
    if avg_ankle_y < nose.y:
        score -= 70 # 这是一个非常危险的行为，扣分最多
        print(f"扣分: 检测到头朝下滑行!")

    # --- 新增规则3: 身体是否面向前方 (通过肩膀的水平宽度判断) ---
    shoulder_width_x = abs(shoulder_l.x - shoulder_r.x)
    body_height_y = abs(avg_hip_y - ((shoulder_l.y + shoulder_r.y)/2))
    
    # 如果肩膀的水平宽度相对于身体高度来说非常小，说明身体是侧向的
    if shoulder_width_x / (body_height_y + 1e-6) < 0.3: # 阈值需要实验调整
        score -= 30
        print(f"扣分: 身体可能为侧向 (肩宽/身高比过小)")

    # --- 原有的站立检测规则 (基于角度) ---
    avg_hip_angle = (calculate_angle(shoulder_l, hip_l, knee_l) + calculate_angle(shoulder_r, hip_r, knee_r)) / 2
    if avg_hip_angle > 130:
        score -= 60
        print(f"扣分: 检测到站立姿势 (髋关节角度 {avg_hip_angle:.2f} > 130度)")
    
    return max(0, score) # 确保分数不为负

# --- 主处理函数 ---
def process_image_for_scoring(image_path, conf_threshold, save_output):
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 '{image_path}'")
        return

    annotated_image = image.copy()
    h, w, _ = image.shape

    # 1. 使用YOLO进行游乐设施检测 (使用可调的置信度阈值)
    print(f"--- 开始YOLOv8检测 (置信度阈值: {conf_threshold}) ---")
    yolo_results = yolo_model(image, conf=conf_threshold)[0]
    
    detected_equipments = []
    if yolo_results.boxes:
        for box in yolo_results.boxes:
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf)
            equipment_box = box.xyxy[0].cpu().numpy()
            detected_equipments.append({'name': class_name, 'box': equipment_box})
            
            x1, y1, x2, y2 = map(int, equipment_box)
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    if not detected_equipments:
        print("YOLOv8未能检测到任何游乐设施。评分流程中止。")
        cv2.imshow("Scoring Result - No Detections", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print(f"YOLOv8检测到 {len(detected_equipments)} 个游乐设施。")

    # 2. 使用MediaPipe进行姿态检测
    print("\n--- 开始MediaPipe姿态检测 ---")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    
    if pose_results.pose_landmarks:
        print("MediaPipe检测到人体姿态。")
        mp_drawing.draw_landmarks(annotated_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 3. 遍历游乐设施和姿态，进行关联和评分
        print("\n--- 开始关联和评分 ---")
        found_interaction = False
        for equipment in detected_equipments:
            if is_pose_in_box(pose_results.pose_landmarks, equipment['box'], (h, w)):
                found_interaction = True
                print(f"检测到有人在 [{equipment['name']}] 上...")
                
                final_score = "N/A"
                if equipment['name'] == 'slide':
                    final_score = calculate_slide_score(pose_results.pose_landmarks)
                
                print(f"玩法得分: {final_score}")
                
                x1, y1, _, _ = map(int, equipment['box'])
                cv2.putText(annotated_image, f"Score: {final_score}", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if not found_interaction:
            print("检测到人和游乐设施，但它们之间没有发生关联（人不在设施上）。")
            
    else:
        print("MediaPipe未能检测到任何人体姿态。评分流程中止。")

    # 4. 显示或保存最终结果图片
    if save_output:
        output_path = "scoring_result.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"\n结果已保存到: {output_path}")
    else:
        cv2.imshow("Scoring Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- 脚本入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='整合YOLOv8和MediaPipe Pose对游乐设施玩法进行评分。')
    parser.add_argument('image_path', type=str, help='要处理的输入图片的路径。')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLOv8物体检测的置信度阈值 (默认: 0.25)。')
    parser.add_argument('--save', action='store_true', help='将结果图片保存到文件而不是显示它。')
    
    args = parser.parse_args()
    
    process_image_for_scoring(args.image_path, args.conf, args.save)
