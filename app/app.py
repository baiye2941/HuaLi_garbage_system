"""
社区垃圾分类识别系统 - Flask 主程序
包含页面路由和接口
"""
import os
import cv2
import json
import time
import uuid
import base64
import threading
import numpy as np
import imageio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from detector import MyDetector, frame_to_base64, ALL_CLASSES, BIN_TYPES

# ===== 基础配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "mp4", "avi", "mov"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 最大200MB
app.secret_key = "my_secret_key_123"

# ===== 模型路径 =====
garbage_model_file = os.path.join(BASE_DIR, "models", "garbege.pt")
fire_model_file    = os.path.join(BASE_DIR, "models", "fire_smoke.pt")
smoke_model_file   = os.path.join(BASE_DIR, "models", "fire_smoke.pt")

# ===== 初始化检测器 =====
detector = MyDetector(
    garbage_model_path=garbage_model_file,
    fire_model_path=fire_model_file,
    smoke_model_path=smoke_model_file,
    conf_threshold=0.5,
    iou_threshold=0.3
)

# ===== 运行数据存储 =====
alert_records = []  # 预警记录列表，最多保存500条

run_stats = {
    "total_count": 0,     # 总检测次数
    "alert_count": 0,     # 总预警次数
    "today_count": 0,     # 今日预警次数
    "class_nums":  {cid: 0 for cid in ALL_CLASSES},  # 各类别检测次数
    "hour_data":   [0] * 24,  # 24小时各小时预警次数
    "start_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

data_lock = threading.Lock()

# ===== 警报冷却管理器 =====
class AlertCooldown:
    """线程安全的警报冷却管理器"""
    # 冷却时间配置（秒）
    COOLDOWN_CONFIG = {
        # 垃圾类别（溢出、散落垃圾）
        "overflow": 15 * 60,   # 垃圾溢出 15 分钟
        "garbage":  15 * 60,   # 散落垃圾 15 分钟
        # 火情/烟雾类别
        "fire":     90,        # 1.5 分钟
        "smoke":    90,        # 1.5 分钟
    }

    def __init__(self):
        self._last_alert_time = {}   # {category_key: timestamp}
        self._lock = threading.Lock()

    def _get_cooldown_seconds(self, class_id: int) -> int:
        """根据类别id获取冷却秒数"""
        # 从 ALL_CLASSES 获取类别名
        class_name = ALL_CLASSES.get(class_id, {}).get("name", "")
        if class_name == "垃圾溢出":
            return self.COOLDOWN_CONFIG["overflow"]
        elif class_name == "散落垃圾":
            return self.COOLDOWN_CONFIG["garbage"]
        elif class_name == "火焰":
            return self.COOLDOWN_CONFIG["fire"]
        elif class_name == "烟雾":
            return self.COOLDOWN_CONFIG["smoke"]
        else:
            # 默认 15 分钟（安全）
            return 15 * 60

    def can_alert(self, class_id: int) -> bool:
        """
        检查该类别的检测结果是否可以触发警报
        返回 True 表示允许报警（未冷却或已过冷却期）
        """
        cooldown = self._get_cooldown_seconds(class_id)
        # 使用类别id作为key，确保不同类别独立冷却
        key = class_id
        with self._lock:
            now = time.time()
            last = self._last_alert_time.get(key, 0)
            if now - last >= cooldown:
                # 更新最后报警时间
                self._last_alert_time[key] = now
                return True
            else:
                remaining = int(cooldown - (now - last))
                print(f"[冷却抑制] 类别 '{ALL_CLASSES[class_id]['name']}' 还需等待 {remaining} 秒")
                return False

    def reset_category(self, class_id: int):
        """手动重置某个类别的冷却（用于测试）"""
        with self._lock:
            self._last_alert_time.pop(class_id, None)

# 全局冷却管理器实例
cooldown_manager = AlertCooldown()


def check_ext(filename):
    """判断文件后缀是否允许上传"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def add_alert_record(scene_info, det_list, img_b64=None):
    """保存一条检测记录到 alert_records"""
    global run_stats
    rec = {
        "id":     str(uuid.uuid4())[:8],
        "time":   scene_info["timestamp"],
        "status": scene_info["status"],
        "types":  scene_info["alert_types"],
        "total":  scene_info["total"],
        "image":  img_b64,
    }
    alert_records.insert(0, rec)
    if len(alert_records) > 500:
        alert_records.pop()

    run_stats["alert_count"] += scene_info["alert_count"]
    run_stats["total_count"] += scene_info["total"]

    cur_hour = datetime.now().hour
    if scene_info["alert_count"] > 0:
        run_stats["hour_data"][cur_hour] += 1
        run_stats["today_count"] += 1

    for det in det_list:
        cid = det["class_id"]
        if cid in run_stats["class_nums"]:
            run_stats["class_nums"][cid] += 1


# ===== 页面路由 =====

@app.route("/")
def page_index():
    return render_template("index.html")


@app.route("/detection")
def page_detection():
    return render_template("detection.html")


@app.route("/alerts")
def page_alerts():
    return render_template("alerts.html")


@app.route("/statistics")
def page_statistics():
    return render_template("statistics.html")


@app.route("/dataset")
def page_dataset():
    return render_template("dataset.html")


# ===== 检测接口 =====

@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    """接收上传的图片文件，进行检测"""
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    f = request.files["file"]
    if not f or not check_ext(f.filename):
        return jsonify({"error": "文件格式不支持"}), 400

    # 读取图片
    img_bytes = f.read()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "图片解析失败"}), 400

    # 检测
    det_list = detector.detect(img)

    # ---- 冷却过滤：只保留允许报警的检测结果 ----
    filtered_det_list = []
    for d in det_list:
        if d["alert"]:
            # 只有需要报警的目标才检查冷却
            if cooldown_manager.can_alert(d["class_id"]):
                filtered_det_list.append(d)
            else:
                # 被冷却抑制：将 alert 标记改为 False，但保留在列表中（前端仍显示框但不触发报警）
                d_copy = d.copy()
                d_copy["alert"] = False
                filtered_det_list.append(d_copy)
        else:
            filtered_det_list.append(d)

    scene_info = detector.check_scene(filtered_det_list)
    result_img = detector.draw_boxes(img, filtered_det_list)
    result_b64 = frame_to_base64(result_img)

    # 有预警才保存记录（注意 scene_info 中的 alert_count 是基于过滤后的）
    if scene_info["alert_count"] > 0:
        add_alert_record(scene_info, filtered_det_list, result_b64)

    return jsonify({
        "success": True,
        "detections": [{
            "class_id":   d["class_id"],
            "class_name": d["class_name"],
            "confidence": d["confidence"],
            "bbox":       d["bbox"],
            "alert":      d["alert"],
            "icon":       d.get("icon", ""),
            "source":     d.get("source_model", ""),
        } for d in filtered_det_list],
        "scene":        scene_info,
        "result_image": result_b64,
    })


@app.route("/api/detect/base64", methods=["POST"])
def api_detect_base64():
    """接收 base64 格式图片，进行检测（摄像头模式使用）"""
    req_data = request.get_json()
    if not req_data or "image" not in req_data:
        return jsonify({"error": "缺少图片数据"}), 400

    b64_str = req_data["image"]
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "图片解析失败"}), 400

    if img is None:
        return jsonify({"error": "图片解析失败"}), 400

    det_list = detector.detect(img)

    # ---- 冷却过滤：只保留允许报警的检测结果 ----
    filtered_det_list = []
    for d in det_list:
        if d["alert"]:
            if cooldown_manager.can_alert(d["class_id"]):
                filtered_det_list.append(d)
            else:
                d_copy = d.copy()
                d_copy["alert"] = False
                filtered_det_list.append(d_copy)
        else:
            filtered_det_list.append(d)

    scene_info = detector.check_scene(filtered_det_list)
    result_img = detector.draw_boxes(img, filtered_det_list)
    result_b64 = frame_to_base64(result_img)

    if scene_info["alert_count"] > 0:
        add_alert_record(scene_info, filtered_det_list, result_b64)

    return jsonify({
        "success": True,
        "detections": [{
            "class_id":   d["class_id"],
            "class_name": d["class_name"],
            "confidence": d["confidence"],
            "bbox":       d["bbox"],
            "alert":      d["alert"],
            "icon":       d.get("icon", ""),
            "source":     d.get("source_model", ""),
        } for d in filtered_det_list],
        "scene":        scene_info,
        "result_image": result_b64,
    })


# ===== 视频检测接口 =====

@app.route("/video")
def page_video():
    """视频检测页面"""
    return render_template("video.html")


@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    f = request.files["file"]
    if not f or not check_ext(f.filename):
        return jsonify({"error": "文件格式不支持"}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    video_exts = {"mp4", "avi", "mov", "mkv", "flv", "wmv"}
    if ext not in video_exts:
        return jsonify({"error": "请上传视频文件"}), 400

    filename = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    input_path = os.path.join(UPLOAD_DIR, unique_name)
    f.save(input_path)

    name, _ = os.path.splitext(unique_name)
    output_filename = f"{name}_detected.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        os.remove(input_path)
        return jsonify({"error": "无法读取视频文件"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or fps > 120:
        fps = 30.0

    # ========== 使用 imageio 创建 H.264 编码器 ==========
    import imageio
    # 确保使用 ffmpeg 后端
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',       # H.264 编码
        quality=8,             # 质量 0-10，越高越好
        pixelformat='yuv420p'  # 兼容浏览器
    )

    skip_frames = int(request.form.get("skip_frames", 3))

    # 跟踪抑制参数
    tracked_objects = []
    alarm_history = []
    IOU_MATCH_THRESH = 0.4
    MEMORY_SECONDS = 3

    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    frame_count = 0
    total_detections = 0
    total_alerts = 0
    alert_frames = 0
    prev_result_img = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            # 跳帧
            if frame_count % skip_frames != 1:
                if prev_result_img is not None:
                    writer.append_data(prev_result_img)
                else:
                    writer.append_data(frame)
                continue

            det_list = detector.detect(frame)

            # 跟踪抑制逻辑
            matched_indices = set()
            new_alerts = []

            for det in det_list:
                best_iou = 0
                best_idx = -1
                for idx, obj in enumerate(tracked_objects):
                    if obj[0] == det["class_id"]:
                        iou = compute_iou(obj[1], det["bbox"])
                        if iou > best_iou and iou >= IOU_MATCH_THRESH:
                            best_iou = iou
                            best_idx = idx
                if best_idx >= 0:
                    matched_indices.add(best_idx)
                    tracked_objects[best_idx] = (det["class_id"], det["bbox"], current_time)
                    det["alert"] = False
                else:
                    found = False
                    for hist in alarm_history:
                        if hist[0] == det["class_id"] and compute_iou(hist[1], det["bbox"]) >= IOU_MATCH_THRESH:
                            if current_time - hist[2] < MEMORY_SECONDS:
                                det["alert"] = False
                                found = True
                                break
                            else:
                                alarm_history.remove(hist)
                                break
                    if not found:
                        new_alerts.append(det)
                        det["alert"] = True

            new_tracked = []
            for idx, obj in enumerate(tracked_objects):
                if idx in matched_indices:
                    new_tracked.append(obj)
                else:
                    alarm_history.append((obj[0], obj[1], current_time))
            tracked_objects = new_tracked

            for det in new_alerts:
                tracked_objects.append((det["class_id"], det["bbox"], current_time))

            alarm_history = [h for h in alarm_history if current_time - h[2] < MEMORY_SECONDS]

            result_img = detector.draw_boxes(frame, det_list)

            frame_alerts = len(new_alerts)
            if frame_alerts > 0:
                alert_frames += 1
                total_alerts += frame_alerts
            total_detections += len(det_list)

            info_text = f"Frame {frame_count}: {len(det_list)} detections, {frame_alerts} alerts"
            cv2.putText(result_img, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 将 OpenCV BGR 转为 RGB（imageio 需要 RGB）
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            writer.append_data(result_img_rgb)
            prev_result_img = result_img_rgb

        writer.close()
        cap.release()
        os.remove(input_path)
        print(f"[视频检测] 完成，输出: {output_path}")
        print(f"[统计] 总帧数: {frame_count}, 报警帧数: {alert_frames}, 总报警次数: {total_alerts}")

    except Exception as e:
        cap.release()
        if 'writer' in locals():
            writer.close()
        if os.path.exists(input_path):
            os.remove(input_path)
        print(f"[视频检测] 错误: {e}")
        return jsonify({"error": f"生成视频失败: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "result_video": output_filename,
        "stats": {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "video_info": f"{width}x{height}, {fps:.1f}fps"
        }
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供视频文件下载，支持浏览器播放"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "文件不存在", 404
    # 根据扩展名设置 MIME
    if filename.endswith('.mp4'):
        mimetype = 'video/mp4'
    else:
        mimetype = 'application/octet-stream'
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mimetype)
    # 允许浏览器缓存和范围请求（支持拖拽进度条）
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'no-cache'
    return response


# ===== 记录接口 =====

@app.route("/api/alerts")
def api_get_alerts():
    """分页查询检测记录"""
    page        = int(request.args.get("page", 1))
    per_page    = int(request.args.get("per_page", 20))
    status_val  = request.args.get("status", "all")

    # 根据状态过滤
    if status_val == "all":
        filtered = alert_records
    elif status_val == "warning":
        filtered = [r for r in alert_records if r["status"] != "normal"]
    else:
        filtered = [r for r in alert_records if r["status"] == status_val]

    total_num = len(filtered)
    start_idx = (page - 1) * per_page
    end_idx   = start_idx + per_page
    # 返回时不包含图片字段（图片单独查询）
    page_data = [{k: v for k, v in r.items() if k != "image"} for r in filtered[start_idx:end_idx]]

    return jsonify({
        "total":    total_num,
        "page":     page,
        "per_page": per_page,
        "records":  page_data,
    })


@app.route("/api/alerts/<rec_id>/image")
def api_get_alert_image(rec_id):
    """获取某条记录的截图"""
    rec = next((r for r in alert_records if r["id"] == rec_id), None)
    if not rec or not rec.get("image"):
        return jsonify({"error": "记录不存在"}), 404
    return jsonify({"image": rec["image"]})


# ===== 统计接口 =====

@app.route("/api/statistics")
def api_get_stats():
    """获取统计数据"""
    cls_list = []
    for cid, num in run_stats["class_nums"].items():
        if num > 0:
            info = ALL_CLASSES[cid]
            cls_list.append({
                "class_id":   cid,
                "class_name": info["name"],
                "count":      num,
                "is_alert":   info["alert"],
            })
    cls_list.sort(key=lambda x: x["count"], reverse=True)

    return jsonify({
        "total_detections":   run_stats["total_count"],
        "total_alerts":       run_stats["alert_count"],
        "today_alerts":       run_stats["today_count"],
        "hourly_alerts":      run_stats["hour_data"],
        "class_stats":        cls_list,
        "start_time":         run_stats["start_time"],
        "alert_record_count": len(alert_records),
    })


@app.route("/api/classes")
def api_get_classes():
    """获取所有类别信息"""
    cls_list = []
    for cid, info in ALL_CLASSES.items():
        cls_list.append({
            "id":    cid,
            "name":  info["name"],
            "en":    info["en"],
            "alert": info["alert"],
            "icon":  info.get("icon", ""),
        })
    return jsonify({"classes": cls_list, "bin_types": BIN_TYPES})


# ===== 系统状态 =====

@app.route("/api/status")
def api_get_status():
    """获取系统当前状态"""
    return jsonify({
        "model_loaded":   any(detector.models_loaded.values()),
        "garbage_model":  detector.models_loaded.get("garbage", False),
        "fire_model":     detector.models_loaded.get("fire", False),
        "smoke_model":    detector.models_loaded.get("smoke", False),
        "mode":           "正常检测" if any(detector.models_loaded.values()) else "演示模式",
        "uptime":         run_stats["start_time"],
        "class_count":    len(ALL_CLASSES),
        "version":        "1.0",
        "name":           "垃圾分类检测系统",
    })


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print("=" * 45)
    print("  垃圾分类检测系统 v1.0")
    print("  访问地址: http://localhost:5000")
    print("=" * 45)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
