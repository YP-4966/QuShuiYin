"""水印和字幕检测模块

提供两种检测方式:
1. 字幕检测 - 检测画面底部的白色/亮色文字
2. 水印检测 - 检测半透明叠加的 logo/文字水印
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class SubtitleDetector:
    """字幕检测器 - 检测画面底部区域的文字"""

    def __init__(
        self,
        bottom_ratio: float = 0.25,
        min_text_height: int = 10,
        max_text_height: int = 80,
        sensitivity: float = 0.7,
    ):
        """
        Args:
            bottom_ratio: 从底部开始检测的区域比例 (0.0-1.0)
            min_text_height: 最小文字高度 (像素)
            max_text_height: 最大文字高度 (像素)
            sensitivity: 灵敏度 (0.0-1.0)，越高越敏感
        """
        self.bottom_ratio = bottom_ratio
        self.min_text_height = min_text_height
        self.max_text_height = max_text_height
        self.sensitivity = sensitivity

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """检测字幕区域，返回二值掩码

        Args:
            frame: BGR格式的图像

        Returns:
            二值掩码 (255=字幕区域, 0=背景)
        """
        h, w = frame.shape[:2]
        # 只处理底部区域
        y_start = int(h * (1 - self.bottom_ratio))
        roi = frame[y_start:, :]

        # 转灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 二值化 - 检测亮色文字 (白色/黄色字幕)
        # 字幕通常是白色或浅色，带深色描边
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 也检测有描边的字幕 (HSV中高亮度高饱和度 -> 黄色字幕)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 白色字幕
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        # 黄色字幕
        yellow_mask = cv2.inRange(hsv, np.array([15, 100, 200]), np.array([35, 255, 255]))
        # 合并
        text_mask = cv2.bitwise_or(bright_mask, white_mask)
        text_mask = cv2.bitwise_or(text_mask, yellow_mask)

        # 形态学操作 - 连接文字笔画
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (15, max(3, self.min_text_height // 2))
        )
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_close)

        # 膨胀以覆盖文字描边
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel_dilate, iterations=2)

        # 过滤 - 只保留符合字幕尺寸的区域
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_mask = np.zeros_like(text_mask)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if self.min_text_height <= ch <= self.max_text_height and cw > 20:
                # 过滤掉太小的噪点
                area_ratio = cv2.contourArea(contour) / (cw * ch)
                if area_ratio > 0.1 * self.sensitivity:
                    cv2.drawContours(result_mask, [contour], -1, 255, -1)

        # 扩展掩码到原始帧大小 (填充到上方)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y_start:, :] = result_mask

        return full_mask


class WatermarkDetector:
    """水印检测器 - 检测图片/视频中的水印"""

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        sensitivity: float = 0.5,
    ):
        """
        Args:
            regions: 水印可能出现的区域列表
                     可选值: "top_left", "top_right", "bottom_left", "bottom_right", "center"
                     默认检查所有角落
            sensitivity: 灵敏度 (0.0-1.0)
        """
        self.regions = regions or [
            "top_left", "top_right", "bottom_left", "bottom_right"
        ]
        self.sensitivity = sensitivity

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """检测水印区域，返回二值掩码

        Args:
            frame: BGR格式的图像

        Returns:
            二值掩码 (255=水印区域, 0=背景)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for region in self.regions:
            region_mask = self._detect_in_region(frame, region)
            mask = cv2.bitwise_or(mask, region_mask)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _detect_in_region(self, frame: np.ndarray, region: str) -> np.ndarray:
        """在指定区域检测水印"""
        h, w = frame.shape[:2]
        margin = 0.15  # 边缘比例

        # 定义区域边界
        y1, y2, x1, x2 = 0, 0, 0, 0
        if "top" in region:
            y1, y2 = 0, int(h * margin)
        if "bottom" in region:
            y1, y2 = int(h * (1 - margin)), h
        if "left" in region:
            x1, x2 = 0, int(w * margin)
        if "right" in region:
            x1, x2 = int(w * (1 - margin)), w
        if region == "center":
            y1, y2 = int(h * 0.3), int(h * 0.7)
            x1, x2 = int(w * 0.3), int(w * 0.7)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # 检测半透明水印的常见特征
        # 方法1: 检测异常的亮度/对比度区域
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 自适应阈值检测
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 方法2: 边缘检测 (水印通常有清晰的边缘)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 合并两种方法
        combined = cv2.bitwise_or(thresh, edges)

        # 放回全图坐标
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = combined

        return full_mask

    def detect_with_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """使用用户提供的手动掩码

        Args:
            frame: BGR格式图像 (用于获取尺寸)
            mask: 二值掩码图像 (255=水印区域)

        Returns:
            处理后的掩码
        """
        # 确保掩码尺寸匹配
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # 膨胀以确保完全覆盖水印边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask
