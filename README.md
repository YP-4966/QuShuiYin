# VSR Web - AI 在线去字幕/水印平台

基于 [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) 核心引擎思想构建的 Web 版在线去字幕工具。支持上传视频或图片，AI 自动检测文字区域并智能修复，处理完成后可直接下载结果。

---

## 功能特性

- **AI 文字检测** - 使用 RapidOCR 深度学习模型自动识别画面中的文字/字幕/水印
- **智能修复** - 自动填补被去除文字的区域，支持两种修复算法
- **视频处理** - 支持完整视频处理，自适应采样检测，保留原始音轨
- **图片处理** - 支持单张图片去字幕/水印
- **实时进度** - 通过 WebSocket 实时推送处理进度
- **暗色主题** - 现代化深色 UI 设计，支持响应式布局
- **拖拽上传** - 支持拖拽或点击上传文件

---

## 技术架构

```
浏览器 (前端)
  │  拖拽上传 / 模式选择 / 实时进度 / 下载结果
  │
  ├── HTTP API (FastAPI)
  │     ├── POST /api/upload        上传文件
  │     ├── POST /api/process/{id}  开始处理
  │     ├── GET  /api/task/{id}     查询状态
  │     ├── GET  /api/download/{id} 下载结果
  │     └── GET  /api/info          系统信息
  │
  ├── WebSocket (实时进度)
  │     └── /ws/progress/{id}       推送处理进度
  │
  └── 处理引擎
        ├── RapidOCR ONNX  →  文字区域检测
        ├── OpenCV Inpaint →  图像修复 (Telea / NS)
        └── FFmpeg          →  音视频合并
```

---

## 项目结构

```
.
├── app.py              # FastAPI 后端主程序
├── static/
│   ├── index.html      # 前端页面
│   ├── style.css       # 暗色主题样式
│   └── app.js          # 前端交互逻辑
├── uploads/            # 上传文件临时存储
├── results/            # 处理结果存储
├── .gitignore
└── README.md
```

---

## 核心模块说明

### 后端 (app.py)

| 模块 | 说明 |
|------|------|
| OCR 引擎 | 基于 RapidOCR ONNX Runtime 的文字检测，自动识别图片/视频帧中的文字区域 |
| 掩码生成 | 根据检测到的文字坐标生成修复掩码，向外扩展 8 像素防止残留 |
| 图像修复 | OpenCV Inpainting 算法，支持 Telea（快速）和 Navier-Stokes（精细）两种模式 |
| 视频处理 | 逐帧处理，自适应采样间隔（≥60fps→4帧，≥30fps→3帧，其他→2帧），复用相邻帧掩码 |
| 音频保留 | 通过 FFmpeg 从原始视频提取音频，合并到处理后的视频中 |
| 任务管理 | 内存任务队列，支持异步处理和状态查询 |
| WebSocket | 实时推送处理进度到前端 |

### 前端

| 文件 | 说明 |
|------|------|
| index.html | 页面结构：顶部导航 + 左侧上传预览 + 右侧控制面板 |
| style.css | 暗色主题（#0f0f13 背景），紫色强调色（#6C5CE7），响应式布局 |
| app.js | 文件上传、模式选择、WebSocket 进度监听、下载触发 |

---

## 快速开始

### 环境要求

- Python 3.10+
- FFmpeg（用于视频音频合并）

### 安装依赖

```bash
pip install fastapi uvicorn python-multipart aiofiles
pip install rapidocr-onnxruntime
pip install opencv-python-headless numpy
```

### 启动服务

```bash
python app.py
```

服务启动后访问 **http://localhost:8080** 即可使用。

---

## 使用方法

1. **上传文件** - 拖拽或点击上传区域，选择视频/图片文件
2. **选择算法** - Telea（速度快）或 NS（质量好）
3. **设置区域**（可选）- 勾选"限定字幕区域"并设置 Y/X 百分比范围，适合只去除底部字幕的场景
4. **开始处理** - 点击"开始处理"按钮，通过进度条查看实时进度
5. **下载结果** - 处理完成后点击"下载结果"

---

## 支持格式

| 类型 | 格式 |
|------|------|
| 视频 | MP4, AVI, MOV, MKV, WebM, FLV, WMV |
| 图片 | JPG, JPEG, PNG, BMP, WebP, TIFF |

文件大小上限：**500MB**

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| POST | `/api/upload` | 上传文件，返回 task_id |
| POST | `/api/process/{task_id}` | 开始处理，参数：method（telea/ns）、sub_area（可选） |
| GET | `/api/task/{task_id}` | 查询任务状态和进度 |
| GET | `/api/download/{task_id}` | 下载处理结果 |
| WS | `/ws/progress/{task_id}` | WebSocket 实时进度推送 |
| GET | `/api/info` | 系统信息（OCR 引擎状态、支持格式等） |

---

## 修复算法对比

| 算法 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| **Telea** | 快 | 一般 | 日常使用，大面积文字去除 |
| **NS** (Navier-Stokes) | 较慢 | 较好 | 需要更精细修复效果时使用 |

---

## 部署说明

### 本地运行

```bash
python app.py
# 默认监听 0.0.0.0:8080
```

### 服务器部署

```bash
# 使用 gunicorn 多进程（推荐）
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080

# 或直接使用 uvicorn
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
```

### Docker（可选）

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn python-multipart aiofiles rapidocr-onnxruntime opencv-python-headless numpy

EXPOSE 8080
CMD ["python", "app.py"]
```

---

## 致谢

- [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) - 核心引擎思想来源
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - 开源 OCR 引擎
- [OpenCV](https://opencv.org/) - 图像处理与修复
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
