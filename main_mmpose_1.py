# 文件名: main_mmpose.py
import cv2
import numpy as np
# 移除 argparse
from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer
from types import SimpleNamespace # 用于创建简单的对象，模拟MediaPipe的landmark结构
import os
import sys
import traceback # 导入 traceback 模块以打印详细错误

# --- 添加获取资源路径的函数 ---
def resource_path(relative_path):
    """ 获取资源的绝对路径，无论是开发环境还是PyInstaller打包后 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 1. 初始化所有模型 ---
print("--- [分析模块] モデルをロード中... ---")
try:
    # 使用 resource_path 确保能找到模型文件
    yolo_model_equipment = YOLO(resource_path('D:/projectenshu/runs/detect/slide_yolov8s_exp12/weights/last.pt'))
    yolo_model_person = YOLO(resource_path('yolov8n.pt'))

    pose_detector = MMPoseInferencer(
        pose2d='td-hm_hrnet-w32_8xb64-210e_coco-256x192'
    )

    print("--- [分析模块] モデルのロードが完了しました。 ---")
except Exception as e:
    print(f"エラー: [分析模块] モデルのロードに失敗しました。エラー内容: {e}")
    raise RuntimeError(f"分析模块模型加载失败: {e}")

# --- 2. 关键点映射表 (无变化) ---
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

# --- 3. 辅助函数 (无变化) ---
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
    if hasattr(pose_landmarks, 'landmark') and len(pose_landmarks.landmark) > max(hip_l_idx, hip_r_idx):
        if pose_landmarks.landmark[hip_l_idx].visibility > 0.1 and pose_landmarks.landmark[hip_r_idx].visibility > 0.1:
            hip_x = (pose_landmarks.landmark[hip_l_idx].x + pose_landmarks.landmark[hip_r_idx].x) / 2
            hip_y = (pose_landmarks.landmark[hip_l_idx].y + pose_landmarks.landmark[hip_r_idx].y) / 2
            pose_x_px = hip_x * w; pose_y_px = hip_y * h
            x1, y1, x2, y2 = box
            return x1 < pose_x_px < x2 and y1 < pose_y_px < y2
    return False

def get_body_tilt_angle(p1, p2):
    if p1.visibility > 0.1 and p2.visibility > 0.1:
        p1 = np.array([p1.x, p1.y]); p2 = np.array([p2.x, p2.y])
        body_vec = p2 - p1; vertical_vec = np.array([0, 1])
        norm_body_vec = np.linalg.norm(body_vec)
        if norm_body_vec > 1e-6:
            cosine_angle = np.dot(body_vec, vertical_vec) / (norm_body_vec * np.linalg.norm(vertical_vec))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
    return 90.0

# --- 4. 评分函数 (无变化) ---
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
    if avg_hip_y >= avg_knee_y and lm.LEFT_HIP.visibility > 0.1 and lm.RIGHT_HIP.visibility > 0.1 and lm.LEFT_KNEE.visibility > 0.1 and lm.RIGHT_KNEE.visibility > 0.1:
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
    if avg_hip_y < avg_shoulder_y and lm.LEFT_HIP.visibility > 0.1 and lm.RIGHT_HIP.visibility > 0.1 and lm.LEFT_SHOULDER.visibility > 0.1 and lm.RIGHT_SHOULDER.visibility > 0.1:
         score -= 60; deductions.append("減点: 上半身が逆さまの状態")
    if avg_ankle_y < avg_hip_y and lm.LEFT_ANKLE.visibility > 0.1 and lm.RIGHT_ANKLE.visibility > 0.1 and lm.LEFT_HIP.visibility > 0.1 and lm.RIGHT_HIP.visibility > 0.1:
         score -= 90; deductions.append("減点: 全身が逆さまの状態")
    if avg_hip_angle > 160 and avg_knee_angle > 160: score -= 40; deductions.append("減点: 立ち姿勢を検出")
    if lm.NOSE.y > avg_hip_y and lm.NOSE.visibility > 0.3 and lm.LEFT_HIP.visibility > 0.1 and lm.RIGHT_HIP.visibility > 0.1:
         score -= 50; deductions.append("減点: 頭が腰より低い状態")
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

    body_tilt_angle = 90.0 # Default value
    if lm.LEFT_SHOULDER.visibility > 0.3 and lm.LEFT_ANKLE.visibility > 0.3:
        body_tilt_angle = get_body_tilt_angle(lm.LEFT_SHOULDER, lm.LEFT_ANKLE)

    is_swinging_high = body_tilt_angle > 40

    if is_standing and is_swinging_high: score -= 80; deductions.append("減点: 立ち乗りかつ大きく揺らす動作")
    elif is_standing: score -= 40; deductions.append("減点: ブランコの上で立ち乗り")

    avg_hip_y = (lm.LEFT_HIP.y + lm.RIGHT_HIP.y) / 2
    if lm.NOSE.y > avg_hip_y and lm.NOSE.visibility > 0.3:
        score -= 50; deductions.append("減点: 頭が腰より低い状態")

    if person_box and equipment_box:
        person_center_y = (person_box[1] + person_box[3]) / 2
        equipment_top_y = equipment_box[1]
        equipment_height = equipment_box[3] - equipment_box[1]
        if equipment_height > 0 and person_center_y < (equipment_top_y + 0.2 * equipment_height):
            score -= 20; deductions.append("減点: ブランコのフレームに登る動作")

    shoulder_width = abs(lm.LEFT_SHOULDER.x - lm.RIGHT_SHOULDER.x)
    if shoulder_width > 1e-6 and lm.LEFT_WRIST.visibility > 0.3 and lm.RIGHT_WRIST.visibility > 0.3:
        is_left_hand_out = (lm.LEFT_SHOULDER.x - lm.LEFT_WRIST.x) > 0.1 * shoulder_width
        is_right_hand_out = (lm.RIGHT_WRIST.x - lm.RIGHT_SHOULDER.x) > 0.1 * shoulder_width
        if is_left_hand_out and is_right_hand_out:
            score -= 15; deductions.append("減点: 腕を広げている（ロープを握っていない可能性）")

    return max(0, score), deductions

# --- 5. 主处理函数 (添加了详细的调试打印) ---
def process_image_with_mmpose(image_frame):
    """
    接收一个 OpenCV 图像帧 (BGR)，进行分析，并返回分析结果列表和带有标注的图像帧。
    """
    print("--- [分析模块] process_image_with_mmpose 関数を開始しました ---")
    if image_frame is None:
        print(f"エラー: [分析模块] 入力フレームが無効です。")
        return [], None

    annotated_image = image_frame.copy(); h, w, _ = image_frame.shape
    results_log = [] # 初始化结果日志

    try:
        # --- 步骤 1: 游乐设施检测 ---
        print("--- [分析模块] ステップ1: 遊具検出を開始... ---")
        yolo_results_equipment = yolo_model_equipment(image_frame, conf=0.25, verbose=False)[0]
        detected_equipments = [{'name': yolo_model_equipment.names[int(box.cls)], 'box': tuple(map(int, box.xyxy[0].cpu().numpy()))} for box in yolo_results_equipment.boxes]
        print(f"--- [分析模块] ステップ1完了: {len(detected_equipments)} 個の遊具を検出しました。")

        # --- 步骤 2: 人体检测 ---
        print("--- [分析模块] ステップ2: 人体検出を開始... ---")
        yolo_results_person = yolo_model_person(image_frame, classes=[0], conf=0.4, verbose=False)[0]
        person_boxes_np = [box.xyxy[0].cpu().numpy() for box in yolo_results_person.boxes]
        print(f"--- [分析模块] ステップ2完了: {len(person_boxes_np)} 人を検出しました。")

        # --- 步骤 3: 使用 MMPose 进行姿态估计 ---
        all_poses_with_boxes = []
        visualization = annotated_image # 默认使用带框的原图作为可视化底图
        print("--- [分析模块] ステップ3: MMPose姿勢推定を開始... ---")
        if len(person_boxes_np) > 0:
            try:
                result_generator = pose_detector(image_frame, bboxes=person_boxes_np, return_vis=True)
                results_data = next(result_generator)
                print("--- [分析模块] MMPose推論が完了しました。結果を解析中... ---")

                predictions = results_data.get('predictions', [])
                # 如果 mmpose 提供了可视化结果，则使用它
                if results_data.get('visualization'):
                    visualization = results_data.get('visualization')[0]
                else:
                    print("--- [分析模块] 警告: MMPoseから可視化結果が得られませんでした。手動で描画します。 ---")
                    visualization = annotated_image # 保持原样

                if predictions:
                    if isinstance(predictions[0], list):
                        person_predictions = predictions[0]
                        print(f"--- [分析模块] MMPoseは {len(person_predictions)} 人の予測結果を返しました。---")
                        for i, person_data_dict in enumerate(person_predictions):
                            keypoints = person_data_dict.get('keypoints')
                            scores = person_data_dict.get('keypoint_scores')

                            if keypoints is not None and scores is not None:
                                print(f"--- [分析模块] {i+1}人目のキーポイントデータを処理中... ---")
                                pose_obj = convert_mmpose_to_mediapipe_format(np.array(keypoints), np.array(scores), (h, w))
                                if i < len(person_boxes_np):
                                    all_poses_with_boxes.append({'pose': pose_obj, 'box': tuple(map(int, person_boxes_np[i]))})
                            else:
                                print(f"--- [分析模块] 警告: {i+1}人目のキーポイントまたはスコアが見つかりません。 ---")
                    else:
                        print(f"--- [分析模块] 警告: 予期しない`predictions`の形式です。Type: {type(predictions[0])} ---")
                else:
                    print("--- [分析模块] 警告: MMPoseの`predictions`リストが空です。 ---")

            except StopIteration:
                 print("--- [分析模块] エラー: MMPose推論ジェネレータから結果を取得できませんでした。 ---")
                 visualization = annotated_image # 保持原样
            except Exception as e:
                 print(f"--- [分析模块] エラー: MMPose姿勢推定中に予期せぬエラーが発生しました: ---")
                 traceback.print_exc() # 打印完整的错误堆栈信息
                 visualization = annotated_image # 保持原样
        else:
            print("--- [分析模块] 人体が検出されなかったため、姿勢推定はスキップされました。 ---")


        if visualization is not None:
            annotated_image = visualization

        print(f"--- [分析模块] ステップ3完了: {len(all_poses_with_boxes)} 個の有効な姿勢を推定しました。")

        # --- 步骤 4 & 5: 关联、评分、可视化 ---
        results_log = [] # 重置/初始化结果日志
        print(f"\n--- [分析模块] ステップ4: 関連付けとスコアリングを開始... ---")
        for eq in detected_equipments:
            persons_on_equipment = []
            for person_data in all_poses_with_boxes:
                if 'pose' in person_data and is_pose_in_box(person_data['pose'], eq['box'], (h, w)):
                    persons_on_equipment.append(person_data)

            if not persons_on_equipment:
                print(f"--- [分析模块] [{eq['name']}] の上には誰もいません。 ---")
                continue

            num_people = len(persons_on_equipment)
            print(f"--- [分析模块] [{eq['name']}] の上で {num_people} 人を検出しました。スコアリング中... ---")

            for person_index, person_data in enumerate(persons_on_equipment):
                if 'pose' not in person_data: continue

                pose = person_data['pose']
                final_score, deductions = "N/A", []

                try:
                    if eq['name'] == 'slide':
                        final_score, deductions = calculate_slide_score(pose)
                    elif eq['name'] == 'climbing':
                        final_score, deductions = calculate_climbing_score(pose, num_people)
                    elif eq['name'] == 'swing' or eq['name'] == 'swimg':
                        final_score, deductions = calculate_swing_score(pose, person_data['box'], eq['box'])
                    else:
                        print(f"警告: [{eq['name']}] に対するスコアリング関数が見つかりません。")
                        final_score = 100
                        deductions = []

                except Exception as e:
                    print(f"エラー: スコアリング関数 ({eq['name']}) の実行中にエラーが発生しました:")
                    traceback.print_exc() # 打印评分函数的详细错误
                    final_score = "Error"
                    deductions = ["スコアリングエラー"]


                print(f"  - [{eq['name']}] の上の{person_index+1}人目のスコア: {final_score}")
                for reason in deductions:
                    print(f"    {reason}")

                # 准备发送到API的数据日志条目
                risk_status = "abnormal" if isinstance(final_score, (int, float)) and final_score < 70 else "normal"
                log_entry = {
                    "equipment_type": eq['name'],
                    "risk_type": risk_status,
                    "score": final_score if isinstance(final_score, (int, float)) else -1,
                    "deductions": deductions,
                    "person_id": f"person_{person_index+1}"
                }
                results_log.append(log_entry)


                # 在图像上绘制分数
                hip_l_idx = PoseLandmark.LEFT_HIP
                hip_r_idx = PoseLandmark.RIGHT_HIP
                if hasattr(pose, 'landmark') and len(pose.landmark) > max(hip_l_idx, hip_r_idx):
                    if pose.landmark[hip_l_idx].visibility > 0.1 and pose.landmark[hip_r_idx].visibility > 0.1:
                        hip_x = int(((pose.landmark[hip_l_idx].x + pose.landmark[hip_r_idx].x) / 2) * w)
                        hip_y = int(((pose.landmark[hip_l_idx].y + pose.landmark[hip_r_idx].y) / 2) * h)
                        score_text = f"Score: {final_score}"
                        (text_width, text_height), baseline = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        text_x = max(0, hip_x - 5)
                        text_y = max(text_height + 10 + baseline, hip_y)
                        box_y1 = text_y - text_height - 5 - baseline
                        box_y2 = text_y + 5
                        if box_y1 < 0: box_y1 = hip_y + baseline + 5; box_y2 = box_y1 + text_height + 5
                        text_y = box_y1 + text_height + baseline

                        cv2.rectangle(annotated_image, (text_x, box_y1), (text_x + text_width + 10, box_y2), (0,0,0), -1)
                        cv2.putText(annotated_image, score_text, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # 在图像上绘制所有游乐设施检测框 (确保它们在最上层)
        for eq in detected_equipments:
            x1, y1, x2, y2 = eq['box']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_image, eq['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        print("--- [分析模块] ステップ4完了: スコアリングと描画が完了しました。 ---")

    except Exception as e:
        print(f"--- [分析模块] エラー: process_image_with_mmpose 関数内で予期せぬエラーが発生しました: ---")
        traceback.print_exc() # 打印主函数的详细错误
        # 即使出错，也尝试返回原始图像和空结果
        return [], image_frame # 返回原始帧，而不是 annotated_image

    # 返回结果列表和最终标注的图像
    print("--- [分析模块] process_image_with_mmpose 関数が正常に終了しました。 ---")
    return results_log, annotated_image

# --- 【移除】脚本入口部分 ---

