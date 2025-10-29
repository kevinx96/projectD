import cv2
import time
import os
import requests # 用于发送 API 请求
import json
from datetime import datetime
from collections import deque # 用于存储分数历史
import threading # 导入线程库
import numpy as np # 【核心修正】导入 NumPy 库

# 从修改后的 main_mmpose 模块导入核心处理函数
try:
    from main_mmpose_1 import process_image_with_mmpose
except ImportError:
    print("エラー: main_mmpose_1.py が同じディレクトリに見つからないか、初期化中にエラーが発生しました。")
    exit()
except RuntimeError as e: # 捕获模型加载失败的异常
    print(f"エラー: 分析モジュールの初期化に失敗しました - {e}")
    exit()
except Exception as e: # 捕获其他可能的导入时错误
    print(f"エラー: main_mmpose_1.py のインポート中に予期せぬエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    exit()


# --- 1. 参数设置 ---
CAPTURE_INTERVAL = 5  # 每隔 5 秒检测一次
SHARED_FOLDER_PATH = r"D:\DetectedImages"
RENDER_API_URL = "https://playground-api-32jz.onrender.com/api/events"
CAMERA_INDEX = 0
SCORE_THRESHOLD = 70
ALERT_THRESHOLD_COUNT = 5
ALERT_SCORE_THRESHOLD = 60
ALERT_WINDOW_SIZE = 20

# --- 2. 初始化 (全局变量) ---
try:
    os.makedirs(SHARED_FOLDER_PATH, exist_ok=True)
    print(f"画像保存フォルダを確認/作成しました: {SHARED_FOLDER_PATH}")
except OSError as e:
    print(f"エラー: 画像保存フォルダ '{SHARED_FOLDER_PATH}' の作成/アクセスに失敗しました - {e}")
    print("共有フォルダの【ローカルパス】が正しいか、書き込み権限があるか確認してください。")
    exit()

# 摄像头
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"エラー: カメラ {CAMERA_INDEX} を開けませんでした。")
    exit()
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"カメラの準備ができました ({width}x{height})。'q'キーを押すと終了します。")

# 线程锁和共享帧
raw_frame_lock = threading.Lock()
latest_raw_frame = None
annotated_frame_lock = threading.Lock()
latest_annotated_frame = None

# 分数历史记录和警报状态
score_history = {}
alert_triggered = {}
# 线程控制
stop_event = threading.Event()


# --- 3. API 发送函数 ---
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

# --- 4. 摄像头读取线程 ---
def camera_thread():
    """后台线程：专门负责高速读取摄像头画面"""
    global latest_raw_frame
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("エラー: カメラからフレームを読み取れませんでした。")
            time.sleep(0.5)
            continue
        
        with raw_frame_lock:
            latest_raw_frame = frame.copy()
        
        # 控制读取帧率，约30fps
        time.sleep(0.03)
    
    print("カメラ読み取りスレッドを終了します。")
    cap.release()

# --- 5. AI分析线程 ---
def analysis_thread():
    """后台线程：专门负责定时进行AI分析、保存和API发送"""
    global latest_annotated_frame, score_history, alert_triggered
    
    while not stop_event.is_set():
        # 严格按照5秒间隔执行
        time.sleep(CAPTURE_INTERVAL) 
        
        current_frame_to_analyze = None
        timestamp_now = datetime.now() # 获取准确的分析开始时间
        timestamp_iso = timestamp_now.isoformat()

        # 从摄像头线程获取最新的一帧原始图像
        with raw_frame_lock:
            if latest_raw_frame is not None:
                current_frame_to_analyze = latest_raw_frame.copy()

        if current_frame_to_analyze is None:
            print("警告: 解析スレッドがカメラからフレームを取得できませんでした。")
            continue
            
        print(f"\n[{timestamp_now.strftime('%Y-%m-%d %H:%M:%S')}] インターバルに到達、フレームを解析します...")

        # --- 调用分析函数 ---
        results_log, annotated_frame = process_image_with_mmpose(current_frame_to_analyze)

        if annotated_frame is None:
            print("警告: 分析モジュールが有効な画像フレームを返しませんでした。")
            annotated_frame = current_frame_to_analyze.copy() # 使用原始帧

        # --- 【核心修改】逻辑变更 ---
        # 1. 无论如何都生成文件名并保存图片
        timestamp_str_file = timestamp_now.strftime("%Y%m%d_%H%M%S")
        base_image_filename = f"capture_{timestamp_str_file}.jpg"
        image_path = os.path.join(SHARED_FOLDER_PATH, base_image_filename)

        save_success = False
        try:
            save_success = cv2.imwrite(image_path, annotated_frame)
            if save_success:
                print(f"結果画像を保存しました: {image_path}")
            else:
                print(f"警告: 結果画像の保存に失敗しました（imwriteがFalseを返しました）。")
        except cv2.error as e:
             print(f"エラー: OpenCVエラーにより結果画像の保存中にエラーが発生しました - {e}")
        except Exception as e:
             print(f"エラー: 結果画像の保存中に予期せぬエラーが発生しました - {e}")

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
                        "camera_id": CAMERA_INDEX,
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
                    "camera_id": CAMERA_INDEX,
                    "equipment_type": "none",
                    "timestamp": timestamp_iso,
                    "risk_type": "normal",
                    "score": 100,
                    "image_filename": base_image_filename,
                    "deductions": []
                }
                send_data_to_api(event_data) # 发送默认日志

        # --- 6. 更新用于主线程显示的标注帧 ---
        # 在帧上添加时间戳
        cv2.putText(annotated_frame, timestamp_now.strftime("%Y-%m-%d %H:%M:%S"), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        with annotated_frame_lock:
            latest_annotated_frame = annotated_frame.copy()

    print("分析スレッドを終了します。")


# --- 7. 主线程 (只负责显示) ---
if __name__ == "__main__":
    
    # 启动后台线程
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    an_thread = threading.Thread(target=analysis_thread, daemon=True)
    
    cam_thread.start()
    an_thread.start()
    
    # 主线程循环，用于显示
    while True:
        display_frame = None
        
        # 优先显示带标注的帧
        with annotated_frame_lock:
            if latest_annotated_frame is not None:
                display_frame = latest_annotated_frame.copy()
        
        # 如果还没有标注帧，则显示原始帧
        if display_frame is None:
            with raw_frame_lock:
                if latest_raw_frame is not None:
                    display_frame = latest_raw_frame.copy()
        
        # 如果仍然没有帧 (例如摄像头刚启动)，显示一个黑色等待画面
        if display_frame is None:
            display_frame = np.zeros((720, 1280, 3), dtype=np.uint8) # 【修正】使用 np
            cv2.putText(display_frame, "カメラを待っています...", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.imshow("Live Analysis - Press 'q' to quit", display_frame)

        # 按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set() # 通知所有线程停止
            break
        
        time.sleep(0.03) # 保持主循环流畅 (约30fps)

    # --- 清理 ---
    print("\nプログラムを終了します。")
    # 线程会自动退出
    cv2.destroyAllWindows()

