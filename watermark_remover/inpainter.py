"""图像修复模块

使用 OpenCV 的 inpainting 算法填充被移除的区域。
支持两种算法:
1. Telea - 快速，适合小区域
2. Navier-Stokes (NS) - 质量更好，适合大面积
"""

import cv2
import numpy as np
from enum import Enum


class InpaintMethod(Enum):
    TELEA = "telea"
    NS = "ns"


class Inpainter:
    """图像修复器"""

    def __init__(
        self,
        method: InpaintMethod = InpaintMethod.NS,
        radius: int = 5,
    ):
        """
        Args:
            method: 修复算法 ("telea" 或 "ns")
            radius: 修复半径，越大越平滑但越慢
        """
        self.method = method
        self.radius = radius

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """使用 inpainting 算法修复图像

        Args:
            frame: BGR格式的原始图像
            mask: 二值掩码 (255=需要修复的区域)

        Returns:
            修复后的图像
        """
        if mask is None or cv2.countNonZero(mask) == 0:
            return frame.copy()

        method_flag = (
            cv2.INPAINT_TELEA
            if self.method == InpaintMethod.TELEA
            else cv2.INPAINT_NS
        )

        result = cv2.inpaint(frame, mask, self.radius, method_flag)
        return result

    def inpaint_multiscale(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """多尺度修复 - 对大面积区域效果更好

        先用较大的半径修复整体，再用较小的半径精修边缘
        """
        if mask is None or cv2.countNonZero(mask) == 0:
            return frame.copy()

        # 第一轮: 大半径粗修复
        result = self.inpaint(frame, mask)

        # 第二轮: 对掩码边缘做精修
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_mask = cv2.erode(mask, kernel, iterations=2)
        edge_mask = cv2.subtract(mask, edge_mask)

        if cv2.countNonZero(edge_mask) > 0:
            small_inpainter = Inpainter(
                method=self.method, radius=max(2, self.radius // 2)
            )
            result = small_inpainter.inpaint(result, edge_mask)

        return result
