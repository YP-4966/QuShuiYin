"""视频处理模块 - 逐帧去除视频中的水印和字幕

使用 ffmpeg 提取帧 → OpenCV 逐帧处理 → ffmpeg 重新合成视频
"""

import cv2
import subprocess
import shutil
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional, Union

from .image_processor import ImageProcessor


class VideoProcessor:
    """视频水印/字幕去除处理器"""

    def __init__(
        self,
        inpaint_method: str = "ns",
        inpaint_radius: int = 5,
        subtitle_bottom_ratio: float = 0.25,
        subtitle_sensitivity: float = 0.7,
        watermark_sensitivity: float = 0.5,
    ):
        self.image_processor = ImageProcessor(
            inpaint_method=inpaint_method,
            inpaint_radius=inpaint_radius,
            subtitle_bottom_ratio=subtitle_bottom_ratio,
            subtitle_sensitivity=subtitle_sensitivity,
            watermark_sensitivity=watermark_sensitivity,
        )

    def remove_subtitle(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        quality: str = "medium",
    ) -> str:
        """去除视频字幕

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            quality: 输出质量 ("low", "medium", "high")

        Returns:
            输出文件路径
        """
        output_path = self._get_output_path(input_path, output_path, "_subtitle_clean")
        return self._process_video(input_path, output_path, "subtitle", quality)

    def remove_watermark(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        quality: str = "medium",
    ) -> str:
        """去除视频水印"""
        output_path = self._get_output_path(input_path, output_path, "_watermark_clean")
        return self._process_video(input_path, output_path, "watermark", quality)

    def remove_both(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        quality: str = "medium",
    ) -> str:
        """同时去除视频字幕和水印"""
        output_path = self._get_output_path(input_path, output_path, "_clean")
        return self._process_video(input_path, output_path, "both", quality)

    def _process_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        mode: str,
        quality: str,
    ) -> str:
        """核心视频处理流程"""
        input_path = str(input_path)
        output_path = str(output_path)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 获取视频信息
        info = self._get_video_info(input_path)
        fps = info["fps"]
        total_frames = info["total_frames"]
        width = info["width"]
        height = info["height"]

        print(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")
        print(f"模式: {mode}, 质量: {quality}")

        # 创建临时目录
        with tempfile.TemporaryDirectory(prefix="wm_remove_") as tmpdir:
            frames_dir = os.path.join(tmpdir, "frames")
            output_frames_dir = os.path.join(tmpdir, "output_frames")
            os.makedirs(frames_dir)
            os.makedirs(output_frames_dir)

            # 1. 提取帧
            print("步骤 1/3: 提取视频帧...")
            self._extract_frames(input_path, frames_dir)

            # 2. 逐帧处理
            print("步骤 2/3: 处理帧...")
            self._process_frames(frames_dir, output_frames_dir, mode)

            # 3. 重新合成视频
            print("步骤 3/3: 合成视频...")
            self._compose_video(
                output_frames_dir, output_path, fps, quality,
                width, height, input_path
            )

        print(f"完成! 输出文件: {output_path}")
        return output_path

    def _get_video_info(self, video_path: str) -> dict:
        """使用 ffmpeg 获取视频信息"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        import json
        info = json.loads(result.stdout)

        video_stream = next(
            s for s in info["streams"] if s["codec_type"] == "video"
        )

        # 解析帧率
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den > 0 else 30.0

        total_frames = int(video_stream.get("nb_frames", 0))
        if total_frames == 0:
            # 从 duration 计算
            duration = float(info["format"].get("duration", 0))
            total_frames = int(duration * fps)

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
        }

    def _extract_frames(self, video_path: str, output_dir: str):
        """使用 ffmpeg 提取帧"""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-qscale:v", "2",
            os.path.join(output_dir, "frame_%06d.png"),
            "-y", "-loglevel", "quiet"
        ]
        subprocess.run(cmd, check=True)

    def _process_frames(self, frames_dir: str, output_dir: str, mode: str):
        """逐帧处理"""
        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        total = len(frame_files)

        for i, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # 根据模式检测和修复
            if mode == "subtitle":
                mask = self.image_processor.subtitle_detector.detect(frame)
            elif mode == "watermark":
                mask = self.image_processor.watermark_detector.detect(frame)
            else:  # both
                sub_mask = self.image_processor.subtitle_detector.detect(frame)
                wm_mask = self.image_processor.watermark_detector.detect(frame)
                mask = cv2.bitwise_or(sub_mask, wm_mask)

            # 修复
            if cv2.countNonZero(mask) > 0:
                result = self.image_processor.inpainter.inpaint(frame, mask)
            else:
                result = frame

            # 保存
            output_path = os.path.join(output_dir, frame_path.name)
            cv2.imwrite(output_path, result)

            # 进度显示
            if (i + 1) % 10 == 0 or (i + 1) == total:
                progress = (i + 1) / total * 100
                print(f"\r  处理进度: {i+1}/{total} ({progress:.1f}%)", end="", flush=True)
        print()

    def _compose_video(
        self, frames_dir: str, output_path: str, fps: float,
        quality: str, width: int, height: int, reference_path: str
    ):
        """使用 ffmpeg 合成视频"""
        # 质量设置
        crf_map = {"low": "35", "medium": "23", "high": "18"}
        crf = crf_map.get(quality, "23")

        # 复制原始音频
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-i", reference_path,  # 第二个输入用于复制音频
            "-c:v", "libx264",
            "-crf", crf,
            "-preset", "medium",
            "-c:a", "copy",  # 复制音频流
            "-map", "0:v:0",  # 使用帧序列的视频
            "-map", "1:a?",  # 使用原始音频 (如果有的话)
            "-shortest",
            "-y", "-loglevel", "warning",
            output_path
        ]
        subprocess.run(cmd, check=True)

    @staticmethod
    def _get_output_path(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]],
        suffix: str,
    ) -> str:
        if output_path:
            return str(output_path)
        p = Path(input_path)
        return str(p.parent / f"{p.stem}{suffix}{p.suffix}")
