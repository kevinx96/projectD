import cv2
import mediapipe as mp
import argparse # 命令行参数处理库

# --- 主函数定义 ---
def main(image_path):
    # 初始化MediaPipe Pose模型
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # 初始化绘图工具
    mp_drawing = mp.solutions.drawing_utils

    # 从指定的路径读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 '{image_path}'。请检查文件路径是否正确。")
        return

    # 获取图片尺寸
    h, w, _ = image.shape
    print(f"成功读取图片: {image_path} (尺寸: {w}x{h})")

    # 将图片从BGR格式转换为RGB格式（MediaPipe需要RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理图片，进行姿态检测
    results = pose.process(image_rgb)

    # 检查是否检测到了姿态
    if results.pose_landmarks:
        print("检测到人体姿态！")

        # 在原始图片上绘制姿态关键点和连接线
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # 显示结果图片
        cv2.imshow('Pose Detection Result', annotated_image)
        print("按任意键关闭结果窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 打印左肩关键点的信息 (示例)
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        print(f"\n左肩坐标 (x, y, z, visibility):")
        print(f"x: {left_shoulder.x:.4f}")
        print(f"y: {left_shoulder.y:.4f}")
        print(f"z: {left_shoulder.z:.4f}")
        print(f"visibility: {left_shoulder.visibility:.4f}")

    else:
        print("未检测到人体姿态。")

    # 释放模型资源
    pose.close()

# --- 脚本入口 ---
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='在一个图像文件中检测人体姿态并进行可视化。')
    parser.add_argument('image_path', type=str, help='要处理的输入图片的路径。')

    args = parser.parse_args()

    # 调用主函数
    main(args.image_path)