import cv2
import time
import os
import requests # 用于发送 API 请求
import json
from datetime import datetime
from collections import deque # 用于存储分数历史
import threading # 导入线程库
import numpy as np # 导入 NumPy 库
import sys        # 【新】用于退出程序

# --- 导入分析模块 (保持不变) ---
try:
    # 路径和文件名按照Canvas中的内容保持不变
    from main_mmpose_1 import process_image_with_mmpose
except ImportError:
    print("エラー: main_mmpose_1.py が同じディレクトリに見つからないか、初期化中にエラーが発生しました。")
    sys.exit()
except RuntimeError as e: # 捕获模型加载失败的异常
    print(f"エラー: 分析モジュールの初期化に失敗しました - {e}")
    sys.exit()
except Exception as e: # 捕获其他可能的导入时错误
    print(f"エラー: main_mmpose_1.py のインポート中に予期せぬエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    sys.exit()


# --- 1. 参数设置 ---
CAPTURE_INTERVAL = 1  # 分析间隔（秒），用于计算跳帧
SHARED_FOLDER_PATH = r"D:\DetectedImages" # 图片保存的【本地】路径
RENDER_API_URL = "https://playground-api-32jz.onrender.com/api/event/submit" # API 提交 URL
LOGICAL_CAMERA_ID = 1  # 摄像头逻辑ID (用于发送 "camera_id")

# --- 【核心修改】指定您的测试视频文件路径 ---
VIDEO_FILE_PATH = r"D:\projectenshu\test\2.mp4" # 【重要】请替换为您的视频文件路径

# 报警逻辑参数 (保持不变)
SCORE_THRESHOLD = 70 
ALERT_THRESHOLD_COUNT = 5
ALERT_SCORE_THRESHOLD = 60
ALERT_WINDOW_SIZE = 20

# --- 2. 初始化 ---
try:
    os.makedirs(SHARED_FOLDER_PATH, exist_ok=True)
    print(f"画像保存フォルダを確認/作成しました: {SHARED_FOLDER_PATH}")
except OSError as e:
    print(f"エラー: 画像保存フォルダ '{SHARED_FOLDER_PATH}' の作成/アクセスに失敗しました - {e}")
    sys.exit()

# 线程锁和共享帧
# (现在只被两个线程使用：分析线程和主显示线程)
annotated_frame_lock = threading.Lock()
latest_annotated_frame = None

# 分数历史
score_history = {}
alert_triggered = {}
# 线程控制
stop_event = threading.Event()

# --- 3. API 发送函数 (保持不变) ---
def send_data_to_api(event_data):
    """将事件数据发送到 Render API"""
    headers = {'Content-Type': 'application/json'}
    try:
        if isinstance(event_data.get("timestamp"), datetime):
             event_data["timestamp"] = event_data["timestamp"].isoformat()
        response = requests.post(RENDER_API_URL, headers=headers, data=json.dumps(event_data), timeout=30)
        response.raise_for_status()
        print(f"APIへのイベントデータ送信成功: {response.status_code}")
        return True
    except requests.exceptions.Timeout:
        print(f"エラー: APIへのイベントデータ送信がタイムアウトしました ({RENDER_API_URL})")
    except requests.exceptions.ConnectionError:
         print(f"エラー: APIサーバーへの接続に失敗しました ({RENDER_API_URL})。")
    except requests.exceptions.RequestException as e:
        print(f"エラー: APIへのイベントデータ送信に失敗しました - {e}")
        try:
            print(f"サーバーからの応答: {e.response.text}")
        except: pass
    except Exception as e:
        print(f"エラー: API送信中に予期せぬエラーが発生しました - {e}")
    return False

# --- 【核心修改】 4. AI分析线程 (现在负责读取视频文件) ---
def analysis_thread():
    """后台线程：负责读取视频文件、定时分析、保存和API发送"""
    global latest_annotated_frame, score_history, alert_triggered
    
    # 在线程内部打开视频文件
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル {VIDEO_FILE_PATH} を開けませんでした。")
        stop_event.set() # 通知主线程退出
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("警告: 動画のFPSが取得できませんでした。デフォルト値 30 を使用します。")
        fps = 30
    
    # 计算需要跳过的帧数
    frames_to_skip = int(fps * CAPTURE_INTERVAL)
    frame_count = 0
    print(f"動画ファイルを開きました (FPS: {fps:.2f})。{CAPTURE_INTERVAL}秒ごとに1フレームを解析します (約{frames_to_skip}フレームごとに1回)。")

    while cap.isOpened() and not stop_event.is_set():
        
        ret, frame = cap.read()
        if not ret:
            print("動画ファイルの終わりに到達しました。")
            break # 视频结束，退出循环
            
        # --- 立即更新显示帧，让主线程可以显示正在跳过的画面 ---
        # (可选，但可以增加“正在快进”的感觉)
        # if frame_count % 10 == 0: # 每10帧更新一次显示
        #    with annotated_frame_lock:
        #        latest_annotated_frame = frame.copy()
        
        frame_count += 1
        
        # --- 【核心逻辑】每隔 (frames_to_skip) 帧进行一次分析 ---
        # (我们使用 frame_count % (frames_to_skip + 1) == 0 来处理第0帧)
        if frame_count % max(1, frames_to_skip + 1) != 0:
            continue # 跳过这一帧
            
        # --- 到达分析间隔，开始处理 ---
        timestamp_now = datetime.now()
        timestamp_iso = timestamp_now.isoformat()
        video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        print(f"\n[{timestamp_now.strftime('%Y-%m-%d %H:%M:%S')}] 動画内時間 {video_time_sec:.2f}秒 のフレームを解析します...")

        # --- 调用分析函数 ---
        results_log, annotated_frame = process_image_with_mmpose(frame)

        if annotated_frame is None:
            print("警告: 分析モジュールが有効な画像フレームを返しませんでした。")
            annotated_frame = frame.copy()

        # 1. 无论如何都生成文件名并保存图片
        timestamp_str_file = timestamp_now.strftime("%Y%m%d_%H%M%S")
        base_image_filename = f"capture_vid_f{frame_count}_{timestamp_str_file}.jpg" # 文件名加入帧数
        image_path = os.path.join(SHARED_FOLDER_PATH, base_image_filename)

        save_success = False
        try:
            save_success = cv2.imwrite(image_path, annotated_frame)
            if save_success:
                print(f"結果画像を保存しました: {image_path}")
            else:
                print(f"警告: 結果画像の保存に失敗しました。")
        except Exception as e:
             print(f"エラー: 結果画像の保存中にエラーが発生しました - {e}")

        # 2. 只有在图片保存成功后，才尝试发送 API 数据
        if not save_success:
            print("画像保存に失敗したため、APIへのデータ送信をスキップしました。")
        else:
            # 3. 根据是否有检测结果，发送不同的 API 数据
            if results_log:
                print(f"有効なプレイスコアが {len(results_log)} 件検出されました。APIに送信します。")
                for res in results_log:
                    equipment = res.get("equipment_type", "unknown")
                    person_id = res.get("person_id", "unknown")
                    score = res.get("score", -1)
                    deductions = res.get("deductions", [])

                    risk_status = "abnormal" if isinstance(score, (int, float)) and score < SCORE_THRESHOLD else "normal"
                    event_data = {
                        "camera_id": LOGICAL_CAMERA_ID, # 使用逻辑ID
                        "equipment_type": equipment,
                        "timestamp": timestamp_iso,
                        "risk_type": risk_status,
                        "score": score,
                        "image_filename": base_image_filename,
                        "deductions": deductions if risk_status == "abnormal" else []
                    }
                    send_data_to_api(event_data) # 发送API

                    # --- 报警逻辑 ---
                    if isinstance(score, (int, float)):
                        history_key = f"{equipment}_{person_id}"
                        if history_key not in score_history:
                            score_history[history_key] = deque(maxlen=ALERT_WINDOW_SIZE)
                            alert_triggered[history_key] = False
                        score_history[history_key].append(score)
                        
                        if not alert_triggered[history_key]:
                            low_score_count = sum(1 for s in score_history[history_key] if s < ALERT_SCORE_THRESHOLD)
                            if low_score_count >= ALERT_THRESHOLD_COUNT:
                                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                print(f"警報: [{equipment}] の上の [{person_id}] が頻繁に危険な動作をしています！")
                                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                alert_triggered[history_key] = True
                        elif alert_triggered[history_key]:
                             if len(score_history[history_key]) >= 5 and all(s >= 80 for s in list(score_history[history_key])[-5:]):
                                 print(f"情報: [{equipment}] の上の [{person_id}] の状態が正常に戻りました。警報をリセットします。")
                                 alert_triggered[history_key] = False
            else:
                print("有効なプレイスコアは検出されませんでした。デフォルトログを送信します。")
                event_data = {
                    "camera_id": LOGICAL_CAMERA_ID,
                    "equipment_type": "none",
                    "timestamp": timestamp_iso,
                    "risk_type": "normal",
                    "score": 100,
                    "image_filename": base_image_filename,
                    "deductions": []
                }
                send_data_to_api(event_data) # 发送默认日志

        # 6. 更新用于主线程显示的标注帧
        with annotated_frame_lock:
            latest_annotated_frame = annotated_frame.copy()

    print("分析スレッドを終了します。")
    cap.release()
    stop_event.set() # 视频结束，通知主线程也退出

# --- 5. 主线程 (只负责显示) ---
if __name__ == "__main__":
    
    # 启动后台分析线程
    an_thread = threading.Thread(target=analysis_thread, daemon=True)
    an_thread.start()
    
    print("--- メインの表示ループを開始します ---")
    # 主线程循环，用于显示
    while not stop_event.is_set(): # 循环直到分析线程发出停止信号
        display_frame = None
        
        with annotated_frame_lock:
            if latest_annotated_frame is not None:
                display_frame = latest_annotated_frame.copy()
        
        if display_frame is None:
            # 视频开始前，显示等待画面
            display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(display_frame, "動画ファイルをロード中...", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.imshow("Video File Analysis - Press 'q' to quit", display_frame)

        if cv2.waitKey(33) & 0xFF == ord('q'): # 稍微等待 (约30fps)
            stop_event.set() # 按 'q' 键，通知分析线程停止
            break
        
        # 如果分析线程已经因为视频播完而停止，主线程也退出
        if not an_thread.is_alive():
            stop_event.set()

    # --- 6. 清理 ---
    print("\nプログラムを終了します。")
    cv2.destroyAllWindows()