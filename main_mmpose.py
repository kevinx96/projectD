# 文件名: main_mmpose.py
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer
from types import SimpleNamespace # 用于创建简单的对象，模拟MediaPipe的landmark结构

# --- 1. 初始化所有模型 ---
print("--- モデルをロード中... ---")
try:
    # 初始化YOLO模型
    # 请确保这里的路径是正确的
    yolo_model_equipment = YOLO('D:/projectenshu/runs/detect/slide_yolov8s_exp12/weights/last.pt')
    yolo_model_person = YOLO('yolov8n.pt') 

    # 初始化 MMPose 推理器
    pose_detector = MMPoseInferencer(
        pose2d='td-hm_hrnet-w32_8xb64-210e_coco-256x192'
    )
    
    print("--- モデルのロードが完了しました。 ---")
except Exception as e:
    print(f"エラー: モデルのロードに失敗しました。エラー内容: {e}")
    exit()

# --- 2. 关键点映射表 (MMPose COCO -> MediaPipe) ---
COCO_TO_MEDIAPIPE_MAP = {
    0: 'NOSE', 1: 'LEFT_EYE', 2: 'RIGHT_EYE', 3: 'LEFT_EAR', 4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER', 6: 'RIGHT_SHOULDER', 7: 'LEFT_ELBOW', 8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST', 10: 'RIGHT_WRIST', 11: 'LEFT_HIP', 12: 'RIGHT_HIP',
    13: 'LEFT_KNEE', 14: 'RIGHT_KNEE', 15: 'LEFT_ANKLE', 16: 'RIGHT_ANKLE'
}
class PoseLandmark:
    pass
for i, name in COCO_TO_MEDIAPIPE_MAP.items():
    setattr(PoseLandmark, name, i)

# --- 3. 辅助函数 ---
def convert_mmpose_to_mediapipe_format(keypoints, scores, img_shape):
    landmarks = [SimpleNamespace(x=0, y=0, visibility=0) for _ in range(len(COCO_TO_MEDIAPIPE_MAP))]
    h, w = img_shape
    for i in range(len(keypoints)):
        if i < len(landmarks):
            landmarks[i].x = keypoints[i][0] / w
            landmarks[i].y = keypoints[i][1] / h
            landmarks[i].visibility = scores[i]
    return SimpleNamespace(landmark=landmarks)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_pose_in_box(pose_landmarks, box, image_shape):
    h, w = image_shape
    hip_l_idx, hip_r_idx = PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP
    if len(pose_landmarks.landmark) > max(hip_l_idx, hip_r_idx):
        hip_x = (pose_landmarks.landmark[hip_l_idx].x + pose_landmarks.landmark[hip_r_idx].x) / 2
        hip_y = (pose_landmarks.landmark[hip_l_idx].y + pose_landmarks.landmark[hip_r_idx].y) / 2
        pose_x_px = hip_x * w; pose_y_px = hip_y * h
        x1, y1, x2, y2 = box
        return x1 < pose_x_px < x2 and y1 < pose_y_px < y2
    return False

def get_body_tilt_angle(p1, p2):
    p1 = np.array([p1.x, p1.y]); p2 = np.array([p2.x, p2.y])
    body_vec = p2 - p1; vertical_vec = np.array([0, 1])
    cosine_angle = np.dot(body_vec, vertical_vec) / (np.linalg.norm(body_vec) * np.linalg.norm(vertical_vec) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- 4. 评分函数 ---
def get_landmarks(pose_landmarks):
    landmarks = pose_landmarks.landmark
    lm = SimpleNamespace()
    for idx, name in COCO_TO_MEDIAPIPE_MAP.items():
        if idx < len(landmarks):
            setattr(lm, name, landmarks[idx])
        else:
            setattr(lm, name, SimpleNamespace(x=0, y=0, visibility=0))
    return lm

def calculate_slide_score(pose_landmarks):
    score, deductions = 100, []
    lm = get_landmarks(pose_landmarks)
    avg_hip_y = (lm.LEFT_HIP.y + lm.RIGHT_HIP.y) / 2
    avg_knee_y = (lm.LEFT_KNEE.y + lm.RIGHT_KNEE.y) / 2
    avg_knee_angle = (calculate_angle(lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE) + 
                      calculate_angle(lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE)) / 2
    if avg_hip_y >= avg_knee_y:
        score -= 20; deductions.append("減点: 不適切な座り姿勢（腰が膝より低い）")
    if avg_knee_angle > 160:
        score -= 15; deductions.append("減点: 不適切な座り姿勢（膝が伸びすぎ）")
    avg_ankle_y = (lm.LEFT_ANKLE.y + lm.RIGHT_ANKLE.y) / 2
    if avg_ankle_y < lm.NOSE.y and lm.NOSE.visibility > 0.3 and lm.LEFT_ANKLE.visibility > 0.3:
        score -= 70; deductions.append("減点: 頭から滑る危険な動作")
    shoulder_width_x = abs(lm.LEFT_SHOULDER.x - lm.RIGHT_SHOULDER.x)
    body_height_y = abs(avg_hip_y - ((lm.LEFT_SHOULDER.y + lm.RIGHT_SHOULDER.y) / 2))
    if body_height_y > 1e-6 and shoulder_width_x / body_height_y < 0.3:
        score -= 30; deductions.append("減点: 体が横向きの状態")
    avg_hip_angle = (calculate_angle(lm.LEFT_SHOULDER, lm.LEFT_HIP, lm.LEFT_KNEE) + 
                     calculate_angle(lm.RIGHT_SHOULDER, lm.RIGHT_HIP, lm.RIGHT_KNEE)) / 2
    if avg_hip_angle > 130:
        score -= 60; deductions.append("減点: 立ち姿勢を検出")
    return max(0, score), deductions

def calculate_climbing_score(pose_landmarks, num_poses_in_box=1):
    score, deductions = 100, []
    lm = get_landmarks(pose_landmarks)
    avg_shoulder_y = (lm.LEFT_SHOULDER.y + lm.RIGHT_SHOULDER.y) / 2
    avg_hip_y = (lm.LEFT_HIP.y + lm.RIGHT_HIP.y) / 2
    avg_ankle_y = (lm.LEFT_ANKLE.y + lm.RIGHT_ANKLE.y) / 2
    avg_hip_angle = (calculate_angle(lm.LEFT_SHOULDER, lm.LEFT_HIP, lm.LEFT_KNEE) + 
                     calculate_angle(lm.RIGHT_SHOULDER, lm.RIGHT_HIP, lm.RIGHT_KNEE)) / 2
    avg_knee_angle = (calculate_angle(lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE) + 
                      calculate_angle(lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE)) / 2
    if avg_hip_y < avg_shoulder_y: score -= 60; deductions.append("減点: 上半身が逆さまの状態")
    if avg_ankle_y < avg_hip_y: score -= 90; deductions.append("減点: 全身が逆さまの状態")
    if avg_hip_angle > 160 and avg_knee_angle > 160: score -= 40; deductions.append("減点: 立ち姿勢を検出")
    if lm.NOSE.y > avg_hip_y: score -= 50; deductions.append("減点: 頭が腰より低い状態")
    if num_poses_in_box > 1: score -= 15; deductions.append(f"減点: 複数人（{num_poses_in_box}人）の重複を検出")
    return max(0, score), deductions

def calculate_swing_score(pose_landmarks, person_box, equipment_box):
    score, deductions = 100, []
    lm = get_landmarks(pose_landmarks)
    avg_hip_angle = (calculate_angle(lm.LEFT_SHOULDER, lm.LEFT_HIP, lm.LEFT_KNEE) + 
                     calculate_angle(lm.RIGHT_SHOULDER, lm.RIGHT_HIP, lm.RIGHT_KNEE)) / 2
    avg_knee_angle = (calculate_angle(lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE) + 
                      calculate_angle(lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE)) / 2
    is_standing = avg_hip_angle > 150 and avg_knee_angle > 150
    body_tilt_angle = get_body_tilt_angle(lm.LEFT_SHOULDER, lm.LEFT_ANKLE)
    is_swinging_high = body_tilt_angle > 40 
    if is_standing and is_swinging_high: score -= 80; deductions.append("減点: 立ち乗りかつ大きく揺らす動作")
    elif is_standing: score -= 40; deductions.append("減点: ブランコの上で立ち乗り")
    avg_hip_y = (lm.LEFT_HIP.y + lm.RIGHT_HIP.y) / 2
    if lm.NOSE.y > avg_hip_y: score -= 50; deductions.append("減点: 頭が腰より低い状態")
    if person_box and equipment_box:
        person_center_y = (person_box[1] + person_box[3]) / 2
        equipment_top_y = equipment_box[1]
        equipment_height = equipment_box[3] - equipment_box[1]
        if equipment_height > 0 and person_center_y < (equipment_top_y + 0.2 * equipment_height):
            score -= 20; deductions.append("減点: ブランコのフレームに登る動作")
    shoulder_width = abs(lm.LEFT_SHOULDER.x - lm.RIGHT_SHOULDER.x)
    if shoulder_width > 1e-6:
        is_left_hand_out = (lm.LEFT_SHOULDER.x - lm.LEFT_WRIST.x) > 0.1 * shoulder_width
        is_right_hand_out = (lm.RIGHT_WRIST.x - lm.RIGHT_SHOULDER.x) > 0.1 * shoulder_width
        if is_left_hand_out and is_right_hand_out:
            score -= 15; deductions.append("減点: 腕を広げている（ロープを握っていない可能性）")
    return max(0, score), deductions

# --- 5. 主处理函数 ---
def process_image_with_mmpose(image_path, conf_threshold, save_output):
    image = cv2.imread(image_path)
    if image is None: 
        print(f"エラー: 画像ファイルを読み込めませんでした '{image_path}'")
        return
    annotated_image = image.copy(); h, w, _ = image.shape

    # 步骤 1: 游乐设施检测
    yolo_results_equipment = yolo_model_equipment(image, conf=conf_threshold, verbose=False)[0]
    detected_equipments = [{'name': yolo_model_equipment.names[int(box.cls)], 'box': tuple(map(int, box.xyxy[0].cpu().numpy()))} for box in yolo_results_equipment.boxes]
    print(f"--- ステップ1: {len(detected_equipments)} 個の遊具を検出しました。")

    # 步骤 2: 人体检测
    yolo_results_person = yolo_model_person(image, classes=[0], conf=0.4, verbose=False)[0] 
    person_boxes_np = [box.xyxy[0].cpu().numpy() for box in yolo_results_person.boxes]
    print(f"--- ステップ2: {len(person_boxes_np)} 人を検出しました。")

    # 步骤 3: 使用 MMPose 进行姿态估计
    all_poses_with_boxes = []
    visualization = None
    if len(person_boxes_np) > 0:
        result_generator = pose_detector(image, bboxes=person_boxes_np, return_vis=True)
        results_data = next(result_generator)
        
        predictions = results_data.get('predictions', [])
        visualization = results_data.get('visualization', [annotated_image])[0]
        
        # --- 【核心修正】---
        if predictions:
            # `predictions` 是一个列表, 它的第一个元素 predictions[0] 也是一个列表，
            # 这个内部列表包含了每个人的姿态数据字典。
            if isinstance(predictions[0], list):
                person_predictions = predictions[0]
                for i, person_data_dict in enumerate(person_predictions):
                    keypoints = person_data_dict.get('keypoints')
                    scores = person_data_dict.get('keypoint_scores')
                    
                    if keypoints is not None and scores is not None:
                        pose_obj = convert_mmpose_to_mediapipe_format(np.array(keypoints), np.array(scores), (h, w))
                        if i < len(person_boxes_np):
                            all_poses_with_boxes.append({'pose': pose_obj, 'box': tuple(map(int, person_boxes_np[i]))})
    
    if visualization is not None:
        annotated_image = visualization

    print(f"--- ステップ3: {len(all_poses_with_boxes)} 個の姿勢を推定しました。")
    
    # 步骤 4 & 5: 关联、评分、可视化
    print(f"\n--- ステップ4: 関連付けとスコアリング ---")
    for eq in detected_equipments:
        persons_on_equipment = []
        for person_data in all_poses_with_boxes:
            if is_pose_in_box(person_data['pose'], eq['box'], (h, w)):
                persons_on_equipment.append(person_data)
        
        if not persons_on_equipment:
            continue
            
        num_people = len(persons_on_equipment)
        print(f"[{eq['name']}] の上で {num_people} 人を検出しました。")

        for person_index, person_data in enumerate(persons_on_equipment):
            pose = person_data['pose']
            final_score, deductions = "N/A", []

            if eq['name'] == 'slide':
                final_score, deductions = calculate_slide_score(pose)
            elif eq['name'] == 'climbing':
                final_score, deductions = calculate_climbing_score(pose, num_people)
            elif eq['name'] == 'swing' or eq['name'] == 'swimg':
                final_score, deductions = calculate_swing_score(pose, person_data['box'], eq['box'])
            
            print(f"  - [{eq['name']}] の上の{person_index+1}人目のスコア: {final_score}")
            for reason in deductions:
                print(f"    {reason}")

            # 在图像上绘制分数
            hip_l_idx = PoseLandmark.LEFT_HIP
            hip_r_idx = PoseLandmark.RIGHT_HIP
            if len(pose.landmark) > max(hip_l_idx, hip_r_idx):
                hip_x = int(((pose.landmark[hip_l_idx].x + pose.landmark[hip_r_idx].x) / 2) * w)
                hip_y = int(((pose.landmark[hip_l_idx].y + pose.landmark[hip_r_idx].y) / 2) * h)
                score_text = f"Score: {final_score}"
                (text_width, text_height), baseline = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_image, (hip_x - 5, hip_y - text_height - 5 - baseline), (hip_x + text_width + 5, hip_y + 5), (0,0,0), -1)
                cv2.putText(annotated_image, score_text, (hip_x, hip_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    # 在图像上绘制所有检测框
    for eq in detected_equipments:
        x1, y1, x2, y2 = eq['box']
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_image, eq['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)


    if save_output:
        output_path = "mmpose_result.jpg"; cv2.imwrite(output_path, annotated_image)
        print(f"\n結果を次のパスに保存しました: {output_path}")
    else:
        cv2.namedWindow("MMPose Result", cv2.WINDOW_NORMAL); cv2.imshow("MMPose Result", annotated_image)
        cv2.waitKey(0); cv2.destroyAllWindows()


# --- 脚本入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8とMMPose(HRNet)を統合し、危険な遊び方をスコアリングします。')
    parser.add_argument('image_path', type=str, help='処理対象の入力画像ファイルのパス。')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLOv8遊具検出の信頼度の閾値（デフォルト: 0.25）。')
    parser.add_argument('--save', action='store_true', help='結果画像を表示せずファイルに保存します。')
    args = parser.parse_args()
    
    process_image_with_mmpose(args.image_path, args.conf, args.save)

