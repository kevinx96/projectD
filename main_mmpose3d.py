import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer
from types import SimpleNamespace

# --- 1. 初始化模型 ---
print("--- 正在加载模型... ---")
try:
    # 1. 游具检测模型
    equipment_model_path = 'D:/projectenshu/runs/detect/slide_yolov8s_exp12/weights/last.pt'
    if not os.path.exists(equipment_model_path):
        print(f"警告: 找不到指定的模型路径: {equipment_model_path}")
        print("将临时使用 'yolov8n.pt'。")
        yolo_model_equipment = YOLO('yolov8n.pt')
    else:
        yolo_model_equipment = YOLO(equipment_model_path)

    # 2. 人物检测模型
    yolo_model_person = YOLO('yolov8n.pt') 
    
    # 3. 3D 姿态估计 (MMPose)
    pose_inferencer = MMPoseInferencer(pose3d='human3d')
    
    print("--- 模型加载完成。 ---")
except Exception as e:
    print(f"错误: 模型加载失败。\n详细信息: {e}")
    exit()

# --- 2. 关键点映射 (H36M 拓扑) ---
H36M_KEYPOINTS = {
    'PELVIS': 0,
    'R_HIP': 1, 'R_KNEE': 2, 'R_ANKLE': 3,
    'L_HIP': 4, 'L_KNEE': 5, 'L_ANKLE': 6,
    'SPINE': 7, 'NECK': 8, 'NOSE': 9, 'HEAD': 10,
    'L_SHOULDER': 11, 'L_ELBOW': 12, 'L_WRIST': 13,
    'R_SHOULDER': 14, 'R_ELBOW': 15, 'R_WRIST': 16
}

class Point3D:
    def __init__(self, x, y, z, score=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.score = score
    def __repr__(self):
        return f"(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"

# --- 3. 3D 几何计算辅助函数 ---

def get_keypoint(keypoints, name):
    idx = H36M_KEYPOINTS.get(name)
    if idx is not None and idx < len(keypoints):
        kp = keypoints[idx]
        score = kp[3] if len(kp) > 3 else 1.0 
        return Point3D(kp[0], kp[1], kp[2], score)
    return Point3D(0, 0, 0, 0)

def calculate_vector_3d(p1, p2):
    return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

def calculate_angle_3d(p1, p2, p3):
    v1 = calculate_vector_3d(p2, p1)
    v2 = calculate_vector_3d(p2, p3)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6: return 0.0
    cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def get_plane_normal(p1, p2, p3):
    v1 = calculate_vector_3d(p1, p2) 
    v2 = calculate_vector_3d(p1, p3)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    return normal / (norm + 1e-6) if norm > 1e-6 else np.array([0, 1, 0])

def calculate_angle_with_vector(vec, target_vec):
    norm_vec = np.linalg.norm(vec)
    norm_target = np.linalg.norm(target_vec)
    if norm_vec < 1e-6 or norm_target < 1e-6: return 0.0
    cosine = np.dot(vec, target_vec) / (norm_vec * norm_target)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

# --- 4. 核心分析逻辑 (基于 Y-Up 坐标系) ---

def auto_fix_coordinate_system(keypoints_3d):
    """
    自动检测并修复坐标系问题。
    针对生成图片可能出现的 Y-Down (像素坐标系) 问题进行翻转。
    策略：
    如果姿态在垂直方向上有明显拉伸（站立/直坐），且头在负半轴，脚在正半轴（相对于Root），
    则判定为 Y-Down 系统，需要翻转 Y 轴。
    """
    nose = get_keypoint(keypoints_3d, 'NOSE')
    l_ankle = get_keypoint(keypoints_3d, 'L_ANKLE')
    r_ankle = get_keypoint(keypoints_3d, 'R_ANKLE')
    pelvis = get_keypoint(keypoints_3d, 'PELVIS') # 通常是 (0,0,0) 但为了保险读取一下
    
    avg_ankle_y = (l_ankle.y + r_ankle.y) / 2.0
    
    # 相对高度
    head_rel_y = nose.y - pelvis.y
    ankle_rel_y = avg_ankle_y - pelvis.y
    
    # 判定逻辑：
    # 1. 垂直跨度大 (说明不是平躺)
    vertical_span = abs(head_rel_y - ankle_rel_y)
    
    # 2. 特征：头为负(低)，脚为正(高) -> 这是典型的 Y-Down (图像坐标系) 特征
    #    而在 Y-Up 系统中，头应该是正(高)，脚是负(低)。
    if vertical_span > 0.3: # 至少有 30cm 的垂直跨度
        if head_rel_y < -0.1 and ankle_rel_y > 0.1:
            print("  [系统检测] 检测到 Y-Down 坐标系 (头负脚正)，正在修正为 Y-Up...")
            # 翻转所有点的 Y 轴
            for kp in keypoints_3d:
                kp[1] = -kp[1]
                
    return keypoints_3d

def analyze_slide_safety_3d(keypoints_3d, person_id=0):
    """
    滑梯安全分析 - MMPose使用Y-Up坐标系
    X: 左右, Y: 上下(向上为正), Z: 前后(朝向相机为正)
    """
    # 1. 自动坐标系修正
    keypoints_3d = auto_fix_coordinate_system(keypoints_3d)

    score, deductions = 100, []
    
    # 获取关键点
    nose = get_keypoint(keypoints_3d, 'NOSE')
    l_shoulder = get_keypoint(keypoints_3d, 'L_SHOULDER')
    r_shoulder = get_keypoint(keypoints_3d, 'R_SHOULDER')
    l_hip = get_keypoint(keypoints_3d, 'L_HIP')
    r_hip = get_keypoint(keypoints_3d, 'R_HIP')
    l_knee = get_keypoint(keypoints_3d, 'L_KNEE')
    r_knee = get_keypoint(keypoints_3d, 'R_KNEE')
    l_ankle = get_keypoint(keypoints_3d, 'L_ANKLE')
    r_ankle = get_keypoint(keypoints_3d, 'R_ANKLE')
    pelvis = get_keypoint(keypoints_3d, 'PELVIS')

    # --- DEBUG LOG ---
    print(f"\n[DEBUG Person {person_id}] 关键坐标 (修正后 Y-Up):")
    print(f"  Pelvis: {pelvis}")
    print(f"  Nose: {nose}")
    print(f"  L_Ankle: {l_ankle}, R_Ankle: {r_ankle}")
    print(f"  L_Shoulder: {l_shoulder}, R_Shoulder: {r_shoulder}")
    
    # === 姿态综合分析 ===
    
    # 计算关键高度
    head_height = nose.y - pelvis.y
    avg_ankle_height = ((l_ankle.y + r_ankle.y) / 2) - pelvis.y
    shoulder_height = ((l_shoulder.y + r_shoulder.y) / 2) - pelvis.y
    
    print(f"\n  高度分析:")
    print(f"    头部相对骨盆: {head_height:.2f}m")
    print(f"    肩部相对骨盆: {shoulder_height:.2f}m")
    print(f"    脚踝相对骨盆: {avg_ankle_height:.2f}m")
    
    # 1. 头朝下判定（最危险）
    # 修正逻辑：只有当头确实比骨盆低很多，且不是因为坐着导致的误差
    if head_height < -0.15: # 稍微放宽一点阈值
        # 再次确认：如果脚在更低的地方，那可能只是弯腰
        if avg_ankle_height < head_height:
             # 脚比头还低，那还是头在上
             pass 
        else:
            score -= 70
            deductions.append(f"⚠️ 严重: 头部朝下滑行 (头低于骨盆 {abs(head_height):.2f}m)")
            
            # 如果脚也在上面，完全倒立
            if avg_ankle_height > 0.1:
                score -= 10
                deductions.append(f"⚠️ 完全倒立姿势")
    
    # 2. 站立判定
    angle_knee_l = calculate_angle_3d(l_hip, l_knee, l_ankle)
    angle_knee_r = calculate_angle_3d(r_hip, r_knee, r_ankle)
    is_legs_straight = angle_knee_l > 150 and angle_knee_r > 150
    
    print(f"\n  腿部分析:")
    print(f"    膝盖角度: L={angle_knee_l:.1f}°, R={angle_knee_r:.1f}°")
    
    # 计算腿部方向
    gravity_down = np.array([0, -1, 0])
    l_leg_vec = calculate_vector_3d(l_hip, l_ankle)
    r_leg_vec = calculate_vector_3d(r_hip, r_ankle)
    
    angle_leg_gravity_l = calculate_angle_with_vector(l_leg_vec, gravity_down)
    angle_leg_gravity_r = calculate_angle_with_vector(r_leg_vec, gravity_down)
    
    print(f"    腿部与重力夹角: L={angle_leg_gravity_l:.1f}°, R={angle_leg_gravity_r:.1f}°")
    
    # 站立判定：腿伸直 且 大致垂直向下 且 脚在骨盆下方
    if is_legs_straight and angle_leg_gravity_l < 45 and angle_leg_gravity_r < 45:
        if avg_ankle_height < -0.2:  # 脚确实在下面
            score -= 60
            deductions.append(f"扣分: 站立滑行 (腿部垂直，角度L:{angle_leg_gravity_l:.1f}°, R:{angle_leg_gravity_r:.1f}°)")
    
    # 3. 身体朝向判定
    torso_normal = get_plane_normal(l_shoulder, r_shoulder, l_hip)
    nx, ny, nz = torso_normal[0], torso_normal[1], torso_normal[2]
    
    print(f"\n  姿态分析:")
    print(f"    躯干法向量: [{nx:.2f}, {ny:.2f}, {nz:.2f}]")
    print(f"    躯干垂直跨度: {shoulder_height:.2f}m")

    abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)
    
    # 侧身判定
    if abs_nx > abs_ny and abs_nx > abs_nz and abs_nx > 0.6:
        score -= 30
        deductions.append(f"扣分: 侧身滑行 (法线X:{nx:.2f})")
    
    # 仰卧/俯卧判定（更严格）
    elif abs_ny > abs_nz and abs_ny > 0.75:
        # 检查躯干是否真的水平
        # 只有当肩部和骨盆高度非常接近时，才判定为躺着
        if abs(shoulder_height) < 0.2:  
            score -= 30
            deductions.append(f"扣分: 仰卧/俯卧姿势 (躯干水平，法线Y:{ny:.2f})")
        elif shoulder_height < 0.45:  # 稍微有点高度，可能是半躺
            # score -= 15 # 暂时移除这个扣分，因为坐姿也容易触发
            deductions.append(f"提示: 身体后仰/前倾 (躯干高度{shoulder_height:.2f}m)")
        else:
            print(f"    -> 法向量Y大，但躯干有高度，可能是模型Z轴压缩导致的坐姿误判，忽略。")

    return max(0, score), deductions

def analyze_climbing_safety_3d(keypoints_3d, person_id=0):
    """攀爬安全分析"""
    keypoints_3d = auto_fix_coordinate_system(keypoints_3d) # 同样应用修复
    score, deductions = 100, []
    nose = get_keypoint(keypoints_3d, 'NOSE')
    pelvis = get_keypoint(keypoints_3d, 'PELVIS')
    l_ankle = get_keypoint(keypoints_3d, 'L_ANKLE')
    r_ankle = get_keypoint(keypoints_3d, 'R_ANKLE')

    # 倒立判定: 头比骨盆低 (Y-Up: Nose.y < Pelvis.y)
    if nose.y < pelvis.y - 0.1:
        score -= 60
        deductions.append(f"扣分: 倒立攀爬 (头低于骨盆)")
        
    # 双脚倒挂: 脚在骨盆上方
    avg_ankle_y = (l_ankle.y + r_ankle.y) / 2
    if avg_ankle_y > pelvis.y + 0.1 and nose.y < pelvis.y:
        score -= 30
        deductions.append("扣分: 双脚倒挂姿势")
        
    return max(0, score), deductions

def analyze_swing_safety_3d(keypoints_3d, person_id=0):
    """秋千安全分析"""
    keypoints_3d = auto_fix_coordinate_system(keypoints_3d) # 同样应用修复
    score, deductions = 100, []
    l_hip = get_keypoint(keypoints_3d, 'L_HIP')
    r_hip = get_keypoint(keypoints_3d, 'R_HIP')
    l_knee = get_keypoint(keypoints_3d, 'L_KNEE')
    r_knee = get_keypoint(keypoints_3d, 'R_KNEE')
    l_ankle = get_keypoint(keypoints_3d, 'L_ANKLE')
    r_ankle = get_keypoint(keypoints_3d, 'R_ANKLE')
    l_shoulder = get_keypoint(keypoints_3d, 'L_SHOULDER')
    r_shoulder = get_keypoint(keypoints_3d, 'R_SHOULDER')
    pelvis = get_keypoint(keypoints_3d, 'PELVIS')
    neck = get_keypoint(keypoints_3d, 'NECK')

    # 站立判定
    angle_knee_l = calculate_angle_3d(l_hip, l_knee, l_ankle)
    angle_knee_r = calculate_angle_3d(r_hip, r_knee, r_ankle)
    angle_hip_l = calculate_angle_3d(l_shoulder, l_hip, l_knee)
    angle_hip_r = calculate_angle_3d(r_shoulder, r_hip, r_knee)
    
    is_standing = (angle_knee_l > 150 and angle_knee_r > 150 and 
                   angle_hip_l > 150 and angle_hip_r > 150)
    if is_standing:
        score -= 50
        deductions.append("扣分: 站立荡秋千")

    # 摇荡幅度: 脊柱与垂直方向的夹角
    spine_vec = calculate_vector_3d(pelvis, neck)  # 骨盆 -> 脖子
    world_up = np.array([0, 1, 0])  # Y-Up坐标系
    
    tilt_angle = calculate_angle_with_vector(spine_vec, world_up)
    
    print(f"  脊柱倾斜角度: {tilt_angle:.1f}°")
    
    if tilt_angle > 60:
        score -= 30
        deductions.append(f"扣分: 摇荡幅度过大 (倾斜{tilt_angle:.1f}°)")
        if is_standing:
            score -= 30
            deductions.append("额外扣分: 站立时的危险摇荡")
            
    return max(0, score), deductions


# --- 5. 主处理流程 ---
def process_image_mmpose_3d_full(image_path, output_path, conf_threshold=0.25):
    img = cv2.imread(image_path)
    if img is None: return False
    h, w, _ = img.shape
    annotated_img = img.copy()

    # 1. 检测游具
    equip_results = yolo_model_equipment(img, conf=conf_threshold, verbose=False)[0]
    detected_equipments = []
    for box in equip_results.boxes:
        cls_id = int(box.cls)
        name = yolo_model_equipment.names[cls_id]
        detected_equipments.append({'name': name, 'box': box.xyxy[0].cpu().numpy()})
    
    print(f"检测到的游具: {len(detected_equipments)}个")

    # 2. 检测人物
    person_results = yolo_model_person(img, classes=[0], conf=0.4, verbose=False)[0]
    person_boxes = [box.xyxy[0].cpu().numpy().tolist() for box in person_results.boxes]
    print(f"检测到的人: {len(person_boxes)}人")

    if len(person_boxes) == 0:
        cv2.imwrite(output_path, annotated_img)
        return True

    # 3. 3D 姿态估计
    try:
        result_generator = pose_inferencer(img, bboxes=person_boxes, return_vis=False)
        result = next(result_generator)
        predictions = result['predictions']
        if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
            predictions = predictions[0]
    except Exception as e:
        print(f"MMPose 推理错误: {e}")
        return False

    # 4. 关联与分析
    for idx, (pred, person_box) in enumerate(zip(predictions, person_boxes)):
        keypoints_3d = pred.get('keypoints')
        if keypoints_3d is None or len(keypoints_3d) == 0:
            continue
        
        # 关联判断
        p_x1, p_y1, p_x2, p_y2 = person_box
        p_center_x = (p_x1 + p_x2) / 2
        p_center_y = (p_y1 + p_y2) / 2
        
        target_equip = None
        for equip in detected_equipments:
            e_x1, e_y1, e_x2, e_y2 = equip['box']
            if e_x1 < p_center_x < e_x2 and e_y1 < p_center_y < e_y2:
                target_equip = equip['name']
                break
        
        if target_equip is None:
            # 如果没关联到，为了测试也默认分析Slide，实际使用请根据需求修改
            # target_equip = 'slide (guessed)' 
            continue

        # 分析
        score, deductions = 100, []
        if 'slide' in target_equip.lower():
            score, deductions = analyze_slide_safety_3d(keypoints_3d, idx)
        elif 'climb' in target_equip.lower():
            score, deductions = analyze_climbing_safety_3d(keypoints_3d, idx)
        elif 'swing' in target_equip.lower() or 'swimg' in target_equip.lower():
            score, deductions = analyze_swing_safety_3d(keypoints_3d, idx)

        # 绘制结果
        draw_x, draw_y = int(person_box[0]), int(person_box[1])
        draw_x = max(0, min(draw_x, w - 10))
        draw_y = max(20, min(draw_y, h - 10))
        color = (0, 255, 0) if score > 80 else (0, 165, 255) if score > 60 else (0, 0, 255)
        
        print(f"\n  Person {idx} on {target_equip}: 安全评分 {score}/100")
        for d in deductions:
            print(f"    - {d}")

        # 绘制评分
        cv2.putText(annotated_img, f"{target_equip}: {score}/100", (draw_x, draw_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制扣分原因
        for i, d in enumerate(deductions):
            cv2.putText(annotated_img, d, (draw_x, draw_y + 15 + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 绘制游具框
    for eq in detected_equipments:
        x1, y1, x2, y2 = map(int, eq['box'])
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(annotated_img, eq['name'], (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imwrite(output_path, annotated_img)
    print(f"\n结果已保存: {output_path}")
    return True

# --- 6. 批处理入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='图片文件夹路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    args = parser.parse_args()
    
    folder = Path(args.folder_path)
    if not folder.exists():
        print(f"错误: 文件夹未找到: {folder}")
        exit()
        
    image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
    image_files = [f for f in image_files if '_result' not in f.name]
    
    print(f"\n找到 {len(image_files)} 张图片")
    
    for img_file in image_files:
        print(f"\n{'='*60}")
        print(f"处理中: {img_file.name}")
        print('='*60)
        
        output_name = f"{img_file.stem}_result_3d{img_file.suffix}"
        process_image_mmpose_3d_full(
            str(img_file), 
            str(folder / output_name), 
            args.conf
        )