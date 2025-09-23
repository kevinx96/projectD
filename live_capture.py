# 文件名: live_capture.py
import cv2
import time
import os
import csv
from datetime import datetime

# 从我们的分析模块中导入核心处理函数
from analyzer import process_image

# --- 1. 参数设置 ---
CAPTURE_INTERVAL = 5  # 每隔5秒检测一次
OUTPUT_DIR = "D:\projectenshu\projectd\output"  # 保存检测图片的文件夹
LOG_FILE = "score_log.csv"  # 保存分数的CSV文件名
CAMERA_INDEX = 0  # 摄像头索引，0通常是默认摄像头

# --- 2. 初始化 ---
# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化CSV日志文件
csv_headers = ["Timestamp", "Image_File", "Equipment", "Person_ID", "Score"]
with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# 打开摄像头
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"エラー: カメラ {CAMERA_INDEX} を開けませんでした。")
    exit()

print("カメラの準備ができました。'q'キーを押すと終了します。")
last_capture_time = 0

# --- 3. 主循环 ---
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("エラー: カメラからフレームを読み取れませんでした。")
        break

    # 检查是否到达5秒间隔
    current_time = time.time()
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        last_capture_time = current_time
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] インターバルに到達、フレームを解析します...")

        # --- 调用分析函数 ---
        results, annotated_frame = process_image(frame)

        # --- 处理分析结果 ---
        if results:
            # 生成带时间戳的文件名
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"capture_{timestamp_str}.jpg"
            image_path = os.path.join(OUTPUT_DIR, image_filename)
            
            # 保存标注后的图片
            cv2.imwrite(image_path, annotated_frame)
            print(f"結果画像を保存しました: {image_path}")

            # 将分数记录到CSV文件
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for res in results:
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        image_filename,
                        res["equipment"],
                        res["person_id"],
                        res["score"]
                    ])
            print(f"スコアを {LOG_FILE} に記録しました。")
        else:
            print("有効なプレイスコアは検出されませんでした。")

        # 在窗口中显示带标注的实时画面
        cv2.imshow("Live Analysis - Press 'q' to quit", annotated_frame)
    else:
        # 在等待间隔期间，显示原始画面
        cv2.imshow("Live Analysis - Press 'q' to quit", frame)


    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. 清理 ---
print("プログラムを終了します。")
cap.release()
cv2.destroyAllWindows()