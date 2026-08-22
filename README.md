# QuShuiYin - 去图片和视频水印/字幕工具

## 项目简介

QuShuiYin 是一个基于 Python 的图像和视频处理工具，用于自动去除图片和视频中的水印和字幕。项目集成了多种先进的图像修复技术，无需外部预训练模型即可获得高质量的修复效果。

### 主要功能

- **去除字幕**：自动检测并移除画面底部的白色/黄色字幕文字
- **去除水印**：检测并移除图片/视频角落的半透明水印（Logo、文字等）
- **同时处理**：一键去除字幕和水印
- **预览功能**：可视化显示检测到的水印/字幕区域
- **手动掩码**：支持用户手动标注水印区域进行精确去除
- **多格式支持**：支持 JPG、PNG、MP4、AVI、MOV 等常见格式

## 项目结构

```
watermark_remover/
├── __init__.py              # 包入口，导出主要类
├── __main__.py              # 支持 python -m 方式运行
├── detector.py              # 水印/字幕检测模块
│   ├── SubtitleDetector     # 字幕检测器
│   └── WatermarkDetector    # 水印检测器
├── inpainter.py             # 基础图像修复模块
│   └── Inpainter            # OpenCV inpainting 封装
├── advanced_inpainter.py    # 增强型图像修复模块（6种技术组合）
│   └── AdvancedInpainter    # 多尺度、边缘引导、纹理合成等
├── image_processor.py       # 图片处理模块
│   └── ImageProcessor       # 图片水印/字幕去除处理器
├── video_processor.py       # 视频处理模块
│   └── VideoProcessor       # 视频水印/字幕去除处理器
└── main.py                  # CLI 命令行入口
```

## 技术原理

### 1. 检测模块 (`detector.py`)

**字幕检测**：
- 使用 HSV 颜色空间分析，检测白色（亮度>200）和黄色字幕
- 形态学操作连接文字笔画
- 轮廓分析过滤噪点，只保留符合字幕尺寸的区域
- 支持可调节的底部检测区域比例（默认 25%）

**水印检测**：
- 在指定区域（四个角落、中心）进行检测
- 结合自适应阈值和 Canny 边缘检测
- 检测半透明叠加的异常亮度/对比度区域
- 支持手动掩码输入

### 2. 修复模块

**基础修复** (`inpainter.py`)：
- OpenCV `cv2.inpaint()` 的两种算法：
  - **Telea**：快速算法，适合小区域
  - **Navier-Stokes (NS)**：基于流体力学，质量更好，适合大面积

**增强型修复** (`advanced_inpainter.py`)：
组合 6 种高级技术，分三个质量级别：

| 技术 | 说明 | 应用级别 |
|------|------|----------|
| **多尺度修复** | 从低分辨率到高分辨率逐步修复，保证全局一致性 | balanced/high |
| **边缘引导** | 检测原图边缘并延伸到修复区域，保持边缘连续性 | balanced/high |
| **纹理合成** | 基于 PatchMatch 从外部区域采样纹理填充 | high |
| **泊松融合** | 无缝克隆使修复区域与周围自然过渡 | high |
| **频域融合** | FFT 频域信息传播，保持频率一致性 | high |
| **导向滤波** | 边缘保持平滑，使用原图作为引导 | balanced/high |

### 3. 视频处理 (`video_processor.py`)

处理流程：
1. **FFmpeg 抽帧**：提取视频所有帧为 PNG 图片
2. **OpenCV 逐帧处理**：对每帧应用检测和修复
3. **FFmpeg 重新合成**：将处理后的帧序列合成为视频，保留原始音频

## 环境要求

- Python 3.10+
- FFmpeg（系统级安装）
- OpenCV 4.0+（带 contrib 模块以支持导向滤波）

## 安装依赖

```bash
pip install opencv-python-headless numpy scikit-image Pillow
```

## 使用方法

### 命令行接口 (CLI)

```bash
# 去除图片字幕
python -m watermark_remover remove-subtitle input.jpg

# 去除图片水印
python -m watermark_remover remove-watermark input.jpg

# 同时去除字幕和水印
python -m watermark_remover remove-both input.jpg

# 预览检测到的区域（红色标记）
python -m watermark_remover preview input.jpg

# 使用手动掩码去除水印（白色区域标记水印位置）
python -m watermark_remover remove-watermark input.jpg --mask mask.png

# 去除视频字幕
python -m watermark_remover remove-subtitle video.mp4
```

### Python API

```python
from watermark_remover import ImageProcessor, VideoProcessor

# 图片处理
processor = ImageProcessor(
    inpaint_method="ns",           # 修复算法: "telea" 或 "ns"
    inpaint_radius=5,              # 修复半径
    inpaint_quality="balanced",    # 修复质量: "fast" / "balanced" / "high"
    subtitle_bottom_ratio=0.25,    # 字幕检测底部区域比例
    subtitle_sensitivity=0.7,      # 字幕检测灵敏度
    watermark_sensitivity=0.5,     # 水印检测灵敏度
)

# 去除字幕
output_path = processor.remove_subtitle("input.jpg", "output.jpg")

# 去除水印（支持手动掩码）
output_path = processor.remove_watermark("input.jpg", "output.jpg", manual_mask_path="mask.png")

# 同时去除
output_path = processor.remove_both("input.jpg", "output.jpg")

# 预览掩码
mask_path = processor.preview_mask("input.jpg", "mask_preview.jpg", detect_type="both")

# 视频处理
video_processor = VideoProcessor(
    inpaint_method="ns",
    inpaint_radius=5,
    inpaint_quality="balanced",
)
output_path = video_processor.remove_subtitle("input.mp4", "output.mp4", quality="medium")
```

## 参数说明

### 修复质量 (`--inpaint-quality`)

| 级别 | 技术组合 | 速度 | 适用场景 |
|------|----------|------|----------|
| `fast` | 基础 NS inpainting | 快 | 批量处理、快速预览 |
| `balanced` | 多尺度 + 边缘引导 + 导向滤波 | 中等 | 日常使用（推荐） |
| `high` | 全部 6 种技术组合 | 慢 | 高质量要求场景 |

### 视频输出质量 (`--quality`)

| 级别 | CRF 值 | 文件大小 | 适用场景 |
|------|--------|----------|----------|
| `low` | 35 | 小 | 快速预览 |
| `medium` | 23 | 中等 | 日常使用（推荐） |
| `high` | 18 | 大 | 高质量保存 |

### 公共参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 修复算法 | `ns` |
| `--radius` | 修复半径 | `5` |
| `--bottom-ratio` | 字幕检测底部区域比例 | `0.25` |
| `--sensitivity` | 检测灵敏度 (0.0-1.0) | `0.7` |

## 支持的文件格式

### 图片
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### 视频
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- FLV (.flv)
- WMV (.wmv)
- WebM (.webm)

## 注意事项

1. **网络依赖**：由于网络环境限制，项目未集成 LaMa 等神经网络模型，使用增强型传统算法替代
2. **视频处理**：视频处理需要较长时间，建议先用小片段测试效果
3. **内存使用**：处理大图片/长视频时可能消耗较多内存，建议关闭其他应用
4. **手动掩码**：对于复杂水印，建议先使用 `preview` 命令查看检测效果，再使用手动掩码精确标注
5. **FFmpeg**：视频处理依赖系统级 FFmpeg，需确保已正确安装

## 开发说明

### 添加新的修复算法

1. 在 `advanced_inpainter.py` 中添加新的修复方法
2. 在 `InpaintQuality` 枚举中添加对应级别
3. 在 `_balanced_inpaint` 或 `_high_quality_inpaint` 中调用新方法

### 添加新的检测方法

1. 在 `detector.py` 中添加新的检测器类
2. 在 `ImageProcessor` 中初始化并使用新检测器

## 许可证

本项目仅供学习和研究使用，请勿用于商业用途。
