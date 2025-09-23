import cv2
import mediapipe as mp
from ultralytics import YOLO
import argparse
import numpy as np

# --- 初始化所有模型 ---
# 1. 加载您自己训练好的YOLOv8模型 (用于检测游乐设施)
yolo_model_equipment = YOLO('D:/projectenshu/runs/detect/slide_yolov8m_exp12/weights/last.pt') 
try:
    # 2. 加载一个标准的YOLOv8模型 (用于检测人)
    yolo_model_person = YOLO('yolov8n.pt') 
except Exception as e:
    print(f"エラー: yolov8n.ptモデルの読み込みに失敗しました。ファイルが存在することを確認してください。エラー内容: {e}")
    exit()
# 3. 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# --- 辅助函数 ---
def calculate_angle(a, b, c):
    # 计算由三个点构成的角度 (b为顶点)
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_pose_in_box(pose_landmarks, box, image_shape):
    # 检查一个姿态是否在边界框内
    h, w = image_shape
    hip_x = (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x + pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
    hip_y = (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    pose_x_px = hip_x * w; pose_y_px = hip_y * h
    x1, y1, x2, y2 = box
    return x1 < pose_x_px < x2 and y1 < pose_y_px < y2

def get_body_tilt_angle(p1, p2):
    # 计算两点连线与垂直线的夹角
    p1 = np.array([p1.x, p1.y]); p2 = np.array([p2.x, p2.y])
    body_vec = p2 - p1
    vertical_vec = np.array([0, 1])
    cosine_angle = np.dot(body_vec, vertical_vec) / (np.linalg.norm(body_vec) * np.linalg.norm(vertical_vec) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# --- 评分函数定义 ---
def calculate_slide_score(pose_landmarks):
    """【核心修改】为滑梯评分的每个扣分项添加了日语输出"""
    score = 100
    landmarks = pose_landmarks.landmark
    shoulder_l, shoulder_r = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_l, hip_r = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_l, knee_r = landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_l, ankle_r = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    
    avg_hip_y, avg_knee_y = (hip_l.y + hip_r.y) / 2, (knee_l.y + knee_r.y) / 2
    avg_knee_angle = (calculate_angle(hip_l, knee_l, ankle_l) + calculate_angle(hip_r, knee_r, ankle_r)) / 2
    
    if avg_hip_y >= avg_knee_y:
        score -= 20
        print(f"減点: 不適切な座り姿勢（腰が膝より低い）")
    if avg_knee_angle > 160:
        score -= 15
        print(f"減点: 不適切な座り姿勢（膝が伸びすぎ）")
    
    avg_ankle_y = (ankle_l.y + ankle_r.y) / 2
    if avg_ankle_y < nose.y:
        score -= 70
        print(f"減点: 頭から滑る危険な動作")
        
    shoulder_width_x = abs(shoulder_l.x - shoulder_r.x)
    body_height_y = abs(avg_hip_y - ((shoulder_l.y + shoulder_r.y) / 2))
    if shoulder_width_x / (body_height_y + 1e-6) < 0.3:
        score -= 30
        print(f"減点: 体が横向きの状態")
        
    avg_hip_angle = (calculate_angle(shoulder_l, hip_l, knee_l) + calculate_angle(shoulder_r, hip_r, knee_r)) / 2
    if avg_hip_angle > 130:
        score -= 60
        print(f"減点: 立ち姿勢を検出")
        
    return max(0, score)

def calculate_climbing_score(pose_landmarks, num_poses_in_box=1):
    score = 100
    landmarks=pose_landmarks.landmark;shoulder_l,shoulder_r=landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER];hip_l,hip_r=landmarks[mp_pose.PoseLandmark.LEFT_HIP],landmarks[mp_pose.PoseLandmark.RIGHT_HIP];knee_l,knee_r=landmarks[mp_pose.PoseLandmark.LEFT_KNEE],landmarks[mp_pose.PoseLandmark.RIGHT_KNEE];ankle_l,ankle_r=landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE];nose=landmarks[mp_pose.PoseLandmark.NOSE];avg_shoulder_y,avg_hip_y=(shoulder_l.y+shoulder_r.y)/2,(hip_l.y+hip_r.y)/2;avg_ankle_y=(ankle_l.y+ankle_r.y)/2;avg_hip_angle=(calculate_angle(shoulder_l,hip_l,knee_l)+calculate_angle(shoulder_r,hip_r,knee_r))/2;avg_knee_angle=(calculate_angle(hip_l,knee_l,ankle_l)+calculate_angle(hip_r,knee_r,ankle_r))/2;
    if avg_hip_y<avg_shoulder_y:score-=60;print(f"減点: 上半身が逆さまの状態")
    if avg_ankle_y<avg_hip_y:score-=90;print(f"減点: 全身が逆さまの状態")
    if avg_hip_angle>160 and avg_knee_angle>160:score-=40;print(f"減点: 立ち姿勢を検出")
    if nose.y>avg_hip_y:score-=50;print(f"減点: 頭が腰より低い状態")
    if num_poses_in_box>1:score-=15;print(f"減点: 複数人（{num_poses_in_box}人）の重複を検出")
    return max(0,score)

def calculate_swing_score(pose_landmarks, person_box, equipment_box):
    score = 100
    landmarks = pose_landmarks.landmark
    shoulder_l, shoulder_r = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_l, hip_r = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_l, knee_r = landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_l, ankle_r = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    wrist_l, wrist_r = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    avg_hip_angle = (calculate_angle(shoulder_l, hip_l, knee_l) + calculate_angle(shoulder_r, hip_r, knee_r)) / 2
    avg_knee_angle = (calculate_angle(hip_l, knee_l, ankle_l) + calculate_angle(hip_r, knee_r, ankle_r)) / 2
    is_standing = avg_hip_angle > 150 and avg_knee_angle > 150
    body_tilt_angle = get_body_tilt_angle(shoulder_l, ankle_l)
    is_swinging_high = body_tilt_angle > 40 
    if is_standing and is_swinging_high:
        score -= 80; print(f"減点: 立ち乗りかつ大きく揺らす動作")
    elif is_standing:
        score -= 40; print(f"減点: ブランコの上で立ち乗り")
    avg_hip_y = (hip_l.y + hip_r.y) / 2
    if nose.y > avg_hip_y:
        score -= 50; print(f"減点: 頭が腰より低い状態")
    if person_box and equipment_box:
        person_center_y = (person_box[1] + person_box[3]) / 2
        equipment_top_y = equipment_box[1]
        equipment_height = equipment_box[3] - equipment_box[1]
        if person_center_y < (equipment_top_y + 0.2 * equipment_height):
            score -= 20; print(f"減点: ブランコのフレームに登る動作")
    shoulder_width = abs(shoulder_l.x - shoulder_r.x)
    is_left_hand_out = (shoulder_l.x - wrist_l.x) > 0.1 * shoulder_width
    is_right_hand_out = (wrist_r.x - shoulder_r.x) > 0.1 * shoulder_width
    if is_left_hand_out and is_right_hand_out:
        score -= 15; print(f"減点: 腕を広げている（ロープを握っていない可能性）")
    return max(0, score)


# --- 主处理函数 ---
def process_image_for_scoring(image_path, conf_threshold, save_output):
    image = cv2.imread(image_path)
    if image is None: 
        print(f"エラー: 画像ファイルを読み込めませんでした '{image_path}'")
        return
    annotated_image = image.copy(); h, w, _ = image.shape
    yolo_results_equipment = yolo_model_equipment(image, conf=conf_threshold)[0]
    detected_equipments = [{'name': yolo_model_equipment.names[int(box.cls)], 'box': tuple(map(int, box.xyxy[0].cpu().numpy()))} for box in yolo_results_equipment.boxes]
    for eq in detected_equipments:
        x1, y1, x2, y2 = eq['box']; label = f"{eq['name']}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2); cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    print(f"--- ステップ1: {len(detected_equipments)} 個の遊具を検出しました。")
    yolo_results_person = yolo_model_person(image, classes=[0], conf=0.4)[0] 
    person_boxes = [tuple(map(int, box.xyxy[0].cpu().numpy())) for box in yolo_results_person.boxes]
    print(f"--- ステップ2: {len(person_boxes)} 人を検出しました。")
    all_poses_with_boxes = []
    for p_box in person_boxes:
        x1, y1, x2, y2 = p_box
        crop_x1, crop_y1, crop_x2, crop_y2 = max(0, x1 - 20), max(0, y1 - 20), min(w, x2 + 20), min(h, y2 + 20)
        person_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if person_crop.size == 0: continue
        results = pose_detector.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            crop_h, crop_w, _ = person_crop.shape
            for lm in results.pose_landmarks.landmark:
                lm.x, lm.y = (lm.x * crop_w + crop_x1) / w, (lm.y * crop_h + crop_y1) / h
            all_poses_with_boxes.append({'pose': results.pose_landmarks, 'box': p_box})
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    print(f"--- ステップ3: {len(all_poses_with_boxes)} 個の姿勢を推定しました。")
    print(f"--- ステップ4: 関連付けとスコアリング ---")
    equipment_occupancy = {eq['name'] + str(i): [] for i, eq in enumerate(detected_equipments)}
    equipment_info = {eq['name'] + str(i): eq for i, eq in enumerate(detected_equipments)}
    for i, eq in enumerate(detected_equipments):
        eq_key = eq['name'] + str(i)
        for person_data in all_poses_with_boxes:
            if is_pose_in_box(person_data['pose'], eq['box'], (h, w)):
                equipment_occupancy[eq_key].append(person_data)
    for eq_key, persons_on_equipment in equipment_occupancy.items():
        if not persons_on_equipment: continue
        num_people = len(persons_on_equipment)
        equipment = equipment_info[eq_key]
        equipment_name = equipment['name']
        print(f"[{equipment_name}] の上で {num_people} 人を検出しました。")
        for person_index, person_data in enumerate(persons_on_equipment):
            pose = person_data['pose']
            p_box = person_data['box']
            eq_box = equipment['box']
            final_score = "N/A"
            if equipment_name == 'slide':
                final_score = calculate_slide_score(pose)
            elif equipment_name == 'climbing':
                final_score = calculate_climbing_score(pose, num_people)
            elif equipment_name == 'swing' or equipment_name == 'swimg':
                final_score = calculate_swing_score(pose, p_box, eq_box)
            print(f"  - [{equipment_name}] の上の{person_index+1}人目のスコア: {final_score}")
            hip_x = int(((pose.landmark[mp_pose.PoseLandmark.LEFT_HIP].x + pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2) * w)
            hip_y = int(((pose.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2) * h)
            score_text = f"Score: {final_score}"
            (text_width, text_height), baseline = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_image, (hip_x, hip_y - text_height - baseline), (hip_x + text_width, hip_y), (0,0,0), -1)
            cv2.putText(annotated_image, score_text, (hip_x, hip_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if save_output:
        output_path = "scoring_result_multi.jpg"; cv2.imwrite(output_path, annotated_image)
        print(f"\n結果を次のパスに保存しました: {output_path}")
    else:
        cv2.namedWindow("Scoring Result", cv2.WINDOW_NORMAL); cv2.imshow("Scoring Result", annotated_image)
        cv2.waitKey(0); cv2.destroyAllWindows()

# --- 脚本入口 ---
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='YOLOv8とMediaPipeを統合し、遊具の危険な遊び方をスコアリングします（複数人対応）。'
    )
    parser.add_argument('image_path', type=str, help='処理対象の入力画像ファイルのパス。')
    parser.add_argument(
        '--model', 
        type=str, 
        default='D:/projectenshu/runs/detect/slide_yolov8s_exp12/weights/last.pt',
        help='独自に訓練したYOLOv8遊具検出モデルのパス（.ptファイル）。'
    )
    parser.add_argument('--conf', type=float, default=0.25, help='YOLOv8物体検出の信頼度の閾値（デフォルト: 0.25）。')
    parser.add_argument('--save', action='store_true', help='結果画像を表示せずファイルに保存します。')
    args = parser.parse_args()
    
    # 在主处理函数前加载所有模型
    print("--- モデルをロード中... ---")
    try:
        yolo_model_equipment = YOLO(args.model)
        yolo_model_person = YOLO('yolov8n.pt') 
        print("モデルのロードが完了しました。")
    except Exception as e:
        print(f"エラー: モデルのロードに失敗しました。パスを確認してください。エラー内容: {e}")
        exit()

    process_image_for_scoring(args.image_path, args.conf, args.save)