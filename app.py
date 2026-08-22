"""
VSR Web - 在线视频/图片去字幕服务
基于 video-subtitle-remover 核心引擎 + FastAPI
"""
import os
import sys
import uuid
import time
import json
import shutil
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import Optional
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── 尝试导入 VSR 核心模块 ──────────────────────────────────
VSR_AVAILABLE = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vsr-core'))
    from backend.tools.constant import InpaintMode
    from backend.config import config
    VSR_AVAILABLE = True
    print("[VSR] Core engine available")
except Exception as e:
    print(f"[VSR] Core engine not available, using built-in engine: {e}")

# ── 尝试导入 RapidOCR ──────────────────────────────────────
RAPID_OCR_AVAILABLE = False
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
    print("[OCR] RapidOCR available")
except Exception as e:
    print(f"[OCR] RapidOCR not available: {e}")

# ── 配置 ────────────────────────────────────────────────────
UPLOAD_DIR = Path(__file__).parent / "uploads"
RESULT_DIR = Path(__file__).parent / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ── 任务管理 ────────────────────────────────────────────────
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(self, task_id: str, filename: str, file_path: str, is_video: bool):
        self.task_id = task_id
        self.filename = filename
        self.file_path = file_path
        self.is_video = is_video
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.message = ""
        self.result_path = None
        self.created_at = time.time()
        self.error = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "is_video": self.is_video,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result_available": self.result_path is not None,
        }

tasks: dict[str, Task] = {}
ws_connections: dict[str, list[WebSocket]] = {}

# ── FastAPI App ─────────────────────────────────────────────
app = FastAPI(title="VSR Web - 在线去字幕", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ── OCR 引擎 ────────────────────────────────────────────────
_ocr_engine = None

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None and RAPID_OCR_AVAILABLE:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def detect_text_regions(image: np.ndarray, sub_area=None) -> list:
    """使用 RapidOCR 检测图片中的文字区域"""
    ocr = get_ocr()
    if ocr is None:
        return []

    result, elapse = ocr(image)
    if result is None or len(result) == 0:
        return []

    regions = []
    h, w = image.shape[:2]
    for item in result:
        box = item[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        box = np.array(box, dtype=np.float32)
        x_min = int(max(0, box[:, 0].min()))
        x_max = int(min(w, box[:, 0].max()))
        y_min = int(max(0, box[:, 1].min()))
        y_max = int(min(h, box[:, 1].max()))

        # 过滤太小的区域
        if (x_max - x_min) < 10 or (y_max - y_min) < 5:
            continue

        # 如果指定了字幕区域，检查是否在区域内
        if sub_area:
            sa_ymin, sa_ymax, sa_xmin, sa_xmax = sub_area
            if not (sa_xmin <= x_min and x_max <= sa_xmax and sa_ymin <= y_min and y_max <= sa_ymax):
                continue

        regions.append((x_min, x_max, y_min, y_max))

    return regions


def create_mask(size: tuple, coords_list: list, expand: int = 8) -> np.ndarray:
    """根据文字坐标生成修复掩码"""
    mask = np.zeros(size, dtype="uint8")
    for xmin, xmax, ymin, ymax in coords_list:
        x1 = max(0, xmin - expand)
        y1 = max(0, ymin - expand)
        x2 = min(size[1], xmax + expand)
        y2 = min(size[0], ymax + expand)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    """单帧修复"""
    if method == "ns":
        return cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    else:
        return cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def process_image(task: Task, method: str = "telea", sub_area: tuple = None):
    """处理单张图片"""
    img = cv2.imread(task.file_path)
    if img is None:
        raise ValueError(f"无法读取图片: {task.file_path}")

    regions = detect_text_regions(img, sub_area)
    if not regions:
        task.message = "未检测到文字区域"
        task.progress = 100
        return img

    mask = create_mask(img.shape[:2], regions)
    result = inpaint_frame(img, mask, method)
    return result


def process_video(task: Task, method: str = "telea", sub_area: tuple = None):
    """处理视频 - 逐帧检测+修复"""
    cap = cv2.VideoCapture(task.file_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {task.file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 采样间隔：根据帧率自适应
    if fps >= 60:
        sample_step = 4
    elif fps >= 30:
        sample_step = 3
    else:
        sample_step = 2

    task.message = f"视频信息: {width}x{height}, {fps:.1f}fps, {frame_count}帧"
    _notify_ws(task)

    # 输出视频
    result_name = f"{task.task_id}_no_sub.mp4"
    result_path = str(RESULT_DIR / result_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    # 缓存上一次检测到的 mask，用于非采样帧
    last_mask = np.zeros((height, width), dtype="uint8")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 采样帧：执行 OCR 检测
        if frame_idx % sample_step == 0:
            regions = detect_text_regions(frame, sub_area)
            if regions:
                last_mask = create_mask((height, width), regions)
            else:
                last_mask = np.zeros((height, width), dtype="uint8")

        # 修复
        if last_mask.any():
            frame = inpaint_frame(frame, last_mask, method)

        writer.write(frame)

        # 更新进度
        task.progress = int((frame_idx / frame_count) * 100)
        if frame_idx % 10 == 0:
            task.message = f"处理中... {frame_idx}/{frame_count} 帧"
            _notify_ws(task)

    cap.release()
    writer.release()

    # 合并原始音频
    _merge_audio(task.file_path, result_path)

    return result_path


def _merge_audio(original_video: str, output_video: str):
    """将原始视频的音频合并到处理后的视频中"""
    import subprocess
    try:
        temp_audio = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        # 提取音频
        subprocess.run([
            'ffmpeg', '-y', '-i', original_video,
            '-acodec', 'copy', '-vn', '-loglevel', 'error',
            temp_audio.name
        ], timeout=120, capture_output=True)

        if os.path.getsize(temp_audio.name) > 0:
            # 合并音视频
            temp_out = output_video + '.tmp.mp4'
            subprocess.run([
                'ffmpeg', '-y', '-i', output_video, '-i', temp_audio.name,
                '-vcodec', 'copy', '-acodec', 'copy', '-loglevel', 'error',
                temp_out
            ], timeout=120, capture_output=True)
            if os.path.exists(temp_out):
                os.replace(temp_out, output_video)

        os.unlink(temp_audio.name)
    except Exception as e:
        print(f"[Audio] 合并音频失败 (不影响视频): {e}")


def _notify_ws(task: Task):
    """通过 WebSocket 推送进度"""
    connections = ws_connections.get(task.task_id, [])
    msg = json.dumps(task.to_dict())
    for ws in connections[:]:
        try:
            asyncio.get_event_loop().create_task(ws.send_text(msg))
        except Exception:
            connections.remove(ws)


# ── API 路由 ────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传视频或图片文件"""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT and ext not in ALLOWED_IMAGE_EXT:
        return JSONResponse({"error": f"不支持的文件格式: {ext}"}, status_code=400)

    task_id = str(uuid.uuid4())[:8]
    is_video = ext in ALLOWED_VIDEO_EXT
    save_path = str(UPLOAD_DIR / f"{task_id}{ext}")

    with open(save_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return JSONResponse({"error": "文件太大，最大支持 500MB"}, status_code=400)
        f.write(content)

    task = Task(task_id, file.filename, save_path, is_video)
    tasks[task_id] = task

    return {
        "task_id": task_id,
        "filename": file.filename,
        "is_video": is_video,
        "file_size": len(content),
    }


@app.post("/api/process/{task_id}")
async def start_process(
    task_id: str,
    method: str = Form("telea"),
    sub_area: Optional[str] = Form(None),
):
    """开始处理任务"""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    if task.status == TaskStatus.PROCESSING:
        return JSONResponse({"error": "任务正在处理中"}, status_code=400)

    # 解析字幕区域
    area = None
    if sub_area:
        try:
            parts = [float(x) for x in sub_area.split(",")]
            if len(parts) == 4:
                area = tuple(int(x) for x in parts)  # (ymin, ymax, xmin, xmax)
        except Exception:
            pass

    task.status = TaskStatus.PROCESSING
    task.progress = 0
    task.message = "开始处理..."
    _notify_ws(task)

    try:
        if task.is_video:
            result_path = process_video(task, method, area)
        else:
            result_img = process_image(task, method, area)
            ext = Path(task.file_path).suffix
            result_name = f"{task_id}_no_sub{ext}"
            result_path = str(RESULT_DIR / result_name)
            cv2.imwrite(result_path, result_img)

        task.result_path = result_path
        task.status = TaskStatus.COMPLETED
        task.progress = 100
        task.message = "处理完成！"
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"处理失败: {e}"
        traceback.print_exc()

    _notify_ws(task)
    return task.to_dict()


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    """获取任务状态"""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return task.to_dict()


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """下载处理结果"""
    task = tasks.get(task_id)
    if not task or not task.result_path:
        return JSONResponse({"error": "结果不存在"}, status_code=404)

    if not os.path.exists(task.result_path):
        return JSONResponse({"error": "结果文件已过期"}, status_code=404)

    ext = Path(task.result_path).suffix
    media_type = "video/mp4" if ext == ".mp4" else f"image/{ext.strip('.')}"
    download_name = f"{Path(task.filename).stem}_去字幕{ext}"

    return FileResponse(
        task.result_path,
        media_type=media_type,
        filename=download_name,
    )


@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket 实时进度推送"""
    await websocket.accept()
    if task_id not in ws_connections:
        ws_connections[task_id] = []
    ws_connections[task_id].append(websocket)

    try:
        while True:
            # 保持连接，接收客户端心跳
            data = await websocket.receive_text()
            if data == "ping":
                task = tasks.get(task_id)
                if task:
                    await websocket.send_text(json.dumps(task.to_dict()))
    except WebSocketDisconnect:
        ws_connections[task_id].remove(websocket)
    except Exception:
        if task_id in ws_connections and websocket in ws_connections[task_id]:
            ws_connections[task_id].remove(websocket)


@app.get("/api/info")
async def get_info():
    """获取系统信息"""
    return {
        "vsr_available": VSR_AVAILABLE,
        "rapid_ocr_available": RAPID_OCR_AVAILABLE,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "supported_video": list(ALLOWED_VIDEO_EXT),
        "supported_image": list(ALLOWED_IMAGE_EXT),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
