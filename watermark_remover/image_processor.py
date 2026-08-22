"""图片处理模块 - 去除图片中的水印和字幕"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .detector import SubtitleDetector, WatermarkDetector
from .inpainter import Inpainter, InpaintMethod


class ImageProcessor:
    """图片水印/字幕去除处理器"""

    def __init__(
        self,
        inpaint_method: str = "ns",
        inpaint_radius: int = 5,
        subtitle_bottom_ratio: float = 0.25,
        subtitle_sensitivity: float = 0.7,
        watermark_sensitivity: float = 0.5,
    ):
        """
        Args:
            inpaint_method: 修复算法 ("telea" 或 "ns")
            inpaint_radius: 修复半径
            subtitle_bottom_ratio: 字幕检测的底部区域比例
            subtitle_sensitivity: 字幕检测灵敏度
            watermark_sensitivity: 水印检测灵敏度
        """
        method = InpaintMethod.NS if inpaint_method == "ns" else InpaintMethod.TELEA
        self.inpainter = Inpainter(method=method, radius=inpaint_radius)
        self.subtitle_detector = SubtitleDetector(
            bottom_ratio=subtitle_bottom_ratio,
            sensitivity=subtitle_sensitivity,
        )
        self.watermark_detector = WatermarkDetector(
            sensitivity=watermark_sensitivity,
        )

    def remove_subtitle(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """去除图片中的字幕

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径 (默认在原文件名后加 _clean)

        Returns:
            输出文件路径
        """
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {input_path}")

        output_path = self._get_output_path(input_path, output_path, "_subtitle_clean")
        return self.remove_subtitle_from_frame(frame, str(output_path))

    def remove_subtitle_from_frame(self, frame: np.ndarray, output_path: str) -> str:
        """从帧中去除字幕并保存"""
        mask = self.subtitle_detector.detect(frame)
        result = self.inpainter.inpaint(frame, mask)
        cv2.imwrite(output_path, result)
        return output_path

    def remove_watermark(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        manual_mask_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """去除图片中的水印

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            manual_mask_path: 手动标注的水印掩码路径 (白色区域为水印)

        Returns:
            输出文件路径
        """
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {input_path}")

        output_path = self._get_output_path(input_path, output_path, "_watermark_clean")

        if manual_mask_path:
            mask = cv2.imread(str(manual_mask_path), cv2.IMREAD_GRAYSCALE)
            mask = self.watermark_detector.detect_with_mask(frame, mask)
        else:
            mask = self.watermark_detector.detect(frame)

        result = self.inpainter.inpaint(frame, mask)
        cv2.imwrite(output_path, result)
        return output_path

    def remove_both(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """同时去除字幕和水印

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径

        Returns:
            输出文件路径
        """
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {input_path}")

        output_path = self._get_output_path(input_path, output_path, "_clean")

        # 检测字幕
        subtitle_mask = self.subtitle_detector.detect(frame)
        # 检测水印
        watermark_mask = self.watermark_detector.detect(frame)

        # 合并掩码
        combined_mask = cv2.bitwise_or(subtitle_mask, watermark_mask)

        # 修复
        result = self.inpainter.inpaint(frame, combined_mask)
        cv2.imwrite(output_path, result)
        return output_path

    def preview_mask(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        detect_type: str = "both",
    ) -> str:
        """预览检测到的掩码区域

        Args:
            input_path: 输入图片路径
            output_path: 掩码输出路径
            detect_type: 检测类型 ("subtitle", "watermark", "both")

        Returns:
            掩码输出路径
        """
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {input_path}")

        output_path = self._get_output_path(input_path, output_path, "_mask")

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if detect_type in ("subtitle", "both"):
            mask = cv2.bitwise_or(mask, self.subtitle_detector.detect(frame))
        if detect_type in ("watermark", "both"):
            mask = cv2.bitwise_or(mask, self.watermark_detector.detect(frame))

        # 生成可视化掩码 (红色叠加)
        vis = frame.copy()
        vis[mask > 0] = [0, 0, 255]  # 红色标记检测区域
        result = cv2.addWeighted(frame, 0.6, vis, 0.4, 0)
        cv2.imwrite(output_path, result)
        return output_path

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
