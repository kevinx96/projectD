import cv2
import time
import os
import requests # 用于发送 API 请求
import json
from datetime import datetime
from collections import deque # 用于存储分数历史
import threading # 导入线程库
import numpy as np # 导入 NumPy 库
import subprocess # 【新】用于运行 FFmpeg
import shlex      # 【新】用于安全地分割命令字符串
import sys        # 【新】用于退出程序

# --- 导入分析模块 (保持不变) ---
try:
    from main_mmpose_1 import process_image_with_mmpose
except ImportError:
    print("エラー: main_mmpose_1.py が同じディレクトリに見つからないか、初期化中にエラーが発生しました。")
    sys.exit()
except RuntimeError as e:
    print(f"エラー: 分析モジュールの初期化に失敗しました - {e}")
    sys.exit()
except Exception as e:
    print(f"エラー: main_mmpose_1.py のインポート中に予期せぬエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    sys.exit()

# --- 1. 参数设置 ---
CAPTURE_INTERVAL = 5  # 每隔 5 秒检测一次
SHARED_FOLDER_PATH = r"D:\DetectedImages" # 图片和HLS片段的【本地】保存路径
CAMERA_INDEX = 1  # [MODIFIED] 摄像头索引，从 0 改为 1，以匹配数据库 PK
SCORE_THRESHOLD = 70 
ALERT_THRESHOLD_COUNT = 5
ALERT_SCORE_THRESHOLD = 60
ALERT_WINDOW_SIZE = 20

# --- 【新】HLS 和 API URL 设置 ---
WEBCAM_NAME = "Integrated Webcam" # 【重要】这个现在仅用于API注册时的命名
HLS_FILENAME = "live.m3u8"
# 事件提交 API
RENDER_API_EVENT_SUBMIT_URL = "https://playground-api-32jz.onrender.com/api/event/submit"
# 【MODIFIED】摄像头注册 API，指向新的公开端点
RENDER_API_CAMERA_URL = "https://playground-api-32jz.onrender.com/api/cameras/register" 


# --- 2. 初始化 ---
try:
    os.makedirs(SHARED_FOLDER_PATH, exist_ok=True)
    print(f"画像保存フォルダを確認/作成しました: {SHARED_FOLDER_PATH}")
except OSError as e:
    print(f"エラー: 画像保存フォルダ '{SHARED_FOLDER_PATH}' の作成/アクセスに失敗しました - {e}")
    sys.exit()

# [MODIFIED] 尝试使用 CAMERA_INDEX - 1 打开摄像头，因为 VideoCapture 索引通常从 0 开始
# 但我们的逻辑 ID 是 1
cap_index = 0 
cap = cv2.VideoCapture(cap_index) 
if not cap.isOpened():
    print(f"エラー: カメラ {cap_index} (逻辑 ID: {CAMERA_INDEX}) を開けませんでした。")
    sys.exit()
else:
    # 立即获取摄像头的实际帧率、宽度和高度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # 某些摄像头可能无法正确报告FPS
        fps = 30 # 使用一个合理的默认值
    print(f"カメラ {cap_index} (逻辑 ID: {CAMERA_INDEX}) の準備ができました ({width}x{height} @ {fps:.2f} FPS)。'q'キーを押すと終了します。")

# 线程锁和共享帧
raw_frame_lock = threading.Lock()
latest_raw_frame = None
annotated_frame_lock = threading.Lock()
latest_annotated_frame = None

# 分数历史
score_history = {}
alert_triggered = {}
# 线程控制
stop_event = threading.Event()

# --- 3. API 发送函数 ---
def send_event_to_api(event_data): # 重命名以示区分
    """将【事件数据】发送到 Render API"""
    headers = {'Content-Type': 'application/json'}
    try:
        if isinstance(event_data.get("timestamp"), datetime):
             event_data["timestamp"] = event_data["timestamp"].isoformat()
        
        # 使用正确的事件提交 URL
        response = requests.post(RENDER_API_EVENT_SUBMIT_URL, headers=headers, data=json.dumps(event_data), timeout=30)
        response.raise_for_status()
        print(f"APIへのイベントデータ送信成功: {response.status_code}")
        return True
    except requests.exceptions.Timeout:
        print(f"エラー: APIへのイベントデータ送信がタイムアウトしました ({RENDER_API_EVENT_SUBMIT_URL})")
    except requests.exceptions.ConnectionError:
         print(f"エラー: APIサーバーへの接続に失敗しました ({RENDER_API_EVENT_SUBMIT_URL})。")
    except requests.exceptions.RequestException as e:
        print(f"エラー: APIへのイベントデータ送信に失敗しました - {e}")
        try:
            print(f"サーバーからの応答: {e.response.text}")
        except: pass
    except Exception as e:
        print(f"エラー: API送信中に予期せぬエラーが発生しました - {e}")
    return False

# --- 【核心修改】4. HLS 推流和摄像头注册 ---
def start_ffmpeg_stream():
    """
    FFmpegをバックグラウンドプロセスとして起動し、HLSストリーミングを開始する
    【修改】现在从 stdin 读取原始视频帧
    """
    hls_output_path = os.path.join(SHARED_FOLDER_PATH, HLS_FILENAME)
    
    # 从全局 cap 对象获取帧大小和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0: frame_rate = 30 # 使用默认值

    command = (
        f'ffmpeg -f rawvideo -pix_fmt bgr24 -s {frame_width}x{frame_height} -r {frame_rate} -i - ' # 从 stdin (-) 读取原始 BGR24 视频
        f'-c:v libx264 -preset ultrafast -tune zerolatency -c:a aac -f hls '
        f'-hls_time 2 -hls_list_size 5 -hls_flags delete_segments '
        f'"{hls_output_path}"'
    )
    
    print(f"--- FFmpeg (stdin) ストリーミングを開始します ---")
    print(f"コマンド: {command}")
    
    try:
        # 【修改】设置 stdin=subprocess.PIPE 以便我们可以向其写入帧
        # 移除 shell=True, 使用 shlex.split 来安全地处理命令 (特别是带引号的路径)
        process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"FFmpegプロセスがPID {process.pid} で起動しました。")
        print(f"HLS出力先: {hls_output_path}")
        return process
    except FileNotFoundError:
        print("エラー: FFmpeg が見つかりません。FFmpegがインストールされ、システムのPATHに含まれていることを確認してください。")
        return None
    except Exception as e:
        print(f"エラー: FFmpegの起動に失敗しました: {e}")
        return None

def register_camera_to_api():
    """起動時にカメラ情報をAPIサーバーに送信する"""
    print(f"--- カメラ情報をAPIサーバー ({RENDER_API_CAMERA_URL}) に登録します ---")
    
    # WEBCAM_NAME 仍然用于在 API 注册时给摄像头一个友好的名字
    camera_data = {
        "name": f"ローカルカメラ - {WEBCAM_NAME} (ID: {CAMERA_INDEX})",
        "hls_filename": HLS_FILENAME
    }
    
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(RENDER_API_CAMERA_URL, headers=headers, data=json.dumps(camera_data), timeout=10)
        response.raise_for_status()
        print(f"カメラ情報の登録に成功しました: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"エラー: カメラ情報の登録に失敗しました - {e}")
        try:
            print(f"サーバーからの応答: {e.response.text}")
        except: pass
        print(f"（注意: Render API側に {RENDER_API_CAMERA_URL} エンドポイントが実装されている必要があります）")

# --- 【核心修改】5. 摄像头读取线程 ---
def camera_thread(ffmpeg_process): # 接收 FFmpeg 进程作为参数
    """
    后台线程：专门负责高速读取摄像头画面
    【修改】同时将帧写入 FFmpeg 的 stdin
    """
    global latest_raw_frame
    
    print("--- カメラ読み取りスレッドを開始しました (FFmpegへの書き込みを含む) ---")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("エラー: カメラからフレームを読み取れませんでした。")
            time.sleep(0.5)
            continue
        
        # 1. 更新帧以供 AI 分析线程使用
        with raw_frame_lock:
            latest_raw_frame = frame.copy()
        
        # 2. 将原始帧写入 FFmpeg 进程的 stdin
        try:
            if ffmpeg_process and ffmpeg_process.stdin:
                ffmpeg_process.stdin.write(frame.tobytes())
        except (IOError, BrokenPipeError):
            print("エラー: FFmpegプロセスへの書き込みに失敗しました。HLSストリーミングが停止した可能性があります。")
            # 可能是 FFmpeg 进程意外终止
            stop_event.set() # 停止所有线程
            break
        except Exception as e:
            print(f"エラー: FFmpegへの書き込み中に予期せぬエラー: {e}")
            break
        
        time.sleep(0.03) # 约30fps
    
    print("カメラ読み取りスレッドを終了します。")
    cap.release()

# --- 6. AI分析线程 (无变化) ---
def analysis_thread():
    """后台线程：专门负责定时进行AI分析、保存和API发送"""
    global latest_annotated_frame, score_history, alert_triggered
    
    while not stop_event.is_set():
        time.sleep(CAPTURE_INTERVAL) # 严格按照5秒间隔执行
        
        current_frame_to_analyze = None
        timestamp_now = datetime.now()
        timestamp_iso = timestamp_now.isoformat()

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
            annotated_frame = current_frame_to_analyze.copy()

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
                        "camera_id": CAMERA_INDEX,
                        "equipment_type": equipment,
                        "timestamp": timestamp_iso,
                        "risk_type": risk_status,
                        "score": score,
                        "image_filename": base_image_filename,
                        "deductions": deductions if risk_status == "abnormal" else []
                    }
                    send_event_to_api(event_data) # 发送API

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
                send_event_to_api(event_data) # 发送默认日志

        # 6. 更新用于主线程显示的标注帧
        with annotated_frame_lock:
            latest_annotated_frame = annotated_frame.copy()

    print("分析スレッドを終了します。")


# --- 7. 主线程 (只负责显示) ---
if __name__ == "__main__":
    
    # --- 【核心修改】启动 FFmpeg ---
    ffmpeg_process = start_ffmpeg_stream()
    if ffmpeg_process is None:
        print("FFmpegの起動に失敗したため、プログラムを終了します。")
        sys.exit()
        
    # --- 【新】注册摄像头到 API ---
    register_camera_to_api()
    
    # 【核心修改】启动后台分析线程，将 ffmpeg 进程传递给 camera_thread
    cam_thread = threading.Thread(target=camera_thread, args=(ffmpeg_process,), daemon=True)
    an_thread = threading.Thread(target=analysis_thread, daemon=True)
    
    cam_thread.start()
    an_thread.start()
    
    print("--- メインの表示ループを開始します ---")
    # 主线程循环，用于显示
    while True:
        display_frame = None
        
        with annotated_frame_lock:
            if latest_annotated_frame is not None:
                display_frame = latest_annotated_frame.copy()
        
        if display_frame is None:
            with raw_frame_lock:
                if latest_raw_frame is not None:
                    display_frame = latest_raw_frame.copy()
        
        if display_frame is None:
            display_frame = np.zeros((height if 'height' in locals() else 480, width if 'width' in locals() else 640, 3), dtype=np.uint8)
            cv2.putText(display_frame, "カメラを待っています...", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.imshow("Live Analysis - Press 'q' to quit", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set() # 通知所有线程停止
            break
        
        time.sleep(0.03) # 保持主循环流畅 (约30fps)

    # --- 8. 清理 ---
    print("\nプログラムを終了します。")
    
    # 停止 FFmpeg 进程
    print(f"FFmpegプロセス (PID {ffmpeg_process.pid}) を停止しています...")
    # 【核心修改】关闭 stdin 管道，让 FFmpeg 知道没有更多数据了
    if ffmpeg_process.stdin:
        try:
            ffmpeg_process.stdin.close()
        except Exception as e:
            print(f"FFmpeg stdin のクローズ中にエラー: {e}")
            
    ffmpeg_process.terminate()
    try:
        ffmpeg_process.wait(timeout=5)
        print("FFmpegプロセスが正常に終了しました。")
    except subprocess.TimeoutExpired:
        print("FFmpegが5秒以内に終了しなかったため、強制終了します。")
        ffmpeg_process.kill()
    except Exception as e:
        print(f"FFmpegプロセスの停止中にエラーが発生しました: {e}")


    cv2.destroyAllWindows()

