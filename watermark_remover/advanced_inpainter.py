"""增强型图像修复模块

组合多种高级修复技术，无需外部模型即可获得高质量修复效果:

1. 多尺度修复 (Multi-scale) - 从粗到细，保证全局一致性
2. 边缘引导修复 (Edge-guided) - 保持边缘连续性
3. 纹理合成 (Texture synthesis) - 基于 PatchMatch 的纹理填充
4. 泊松融合 (Poisson blending) - 无缝过渡
5. 频域修复 (Frequency-domain) - FFT 频域信息传播
6. 置信度传播 (Confidence propagation) - 渐进式修复
7. 导向滤波 (Guided filter) - 边缘保持平滑
"""

import cv2
import numpy as np
from typing import Optional
from enum import Enum


class InpaintQuality(Enum):
    FAST = "fast"         # 快速: 基础 NS inpainting
    BALANCED = "balanced" # 均衡: 多尺度 + 边缘引导
    HIGH = "high"         # 高质量: 全部技术组合


class AdvancedInpainter:
    """增强型图像修复器"""

    def __init__(self, quality: InpaintQuality = InpaintQuality.BALANCED):
        self.quality = quality

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """根据质量等级选择修复策略"""
        if mask is None or cv2.countNonZero(mask) == 0:
            return frame.copy()

        if self.quality == InpaintQuality.FAST:
            return self._fast_inpaint(frame, mask)
        elif self.quality == InpaintQuality.BALANCED:
            return self._balanced_inpaint(frame, mask)
        else:
            return self._high_quality_inpaint(frame, mask)

    def _fast_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """快速模式: OpenCV NS inpainting"""
        return cv2.inpaint(frame, mask, 5, cv2.INPAINT_NS)

    def _balanced_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """均衡模式: 多尺度 + 边缘引导 + 导向滤波"""
        # 1. 多尺度修复
        result = self._multiscale_inpaint(frame, mask, scales=2)
        # 2. 边缘引导精修
        result = self._edge_guided_refine(result, frame, mask)
        # 3. 导向滤波平滑
        result = self._guided_filter_smooth(result, frame, mask)
        return result

    def _high_quality_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """高质量模式: 全部技术组合"""
        # 1. 多尺度修复 (3级)
        result = self._multiscale_inpaint(frame, mask, scales=3)
        # 2. 边缘引导精修
        result = self._edge_guided_refine(result, frame, mask)
        # 3. 纹理合成
        result = self._texture_synthesis(result, frame, mask)
        # 4. 泊松融合
        result = self._poisson_blend(result, frame, mask)
        # 5. 频域补偿
        result = self._frequency_blend(result, frame, mask)
        # 6. 导向滤波最终平滑
        result = self._guided_filter_smooth(result, frame, mask)
        return result

    # ==================== 核心修复技术 ====================

    def _multiscale_inpaint(
        self, frame: np.ndarray, mask: np.ndarray, scales: int = 2
    ) -> np.ndarray:
        """多尺度修复 - 从低分辨率到高分辨率逐步修复

        低分辨率修复建立全局结构，高分辨率修复补充细节。
        """
        h, w = frame.shape[:2]

        # 构建图像金字塔
        frames_pyr = [frame]
        masks_pyr = [mask]
        for i in range(scales):
            frame_small = cv2.pyrDown(frames_pyr[-1])
            mask_small = cv2.pyrDown(masks_pyr[-1])
            _, mask_small = cv2.threshold(mask_small, 127, 255, cv2.THRESH_BINARY)
            frames_pyr.append(frame_small)
            masks_pyr.append(mask_small)

        # 从最底层开始修复
        result = cv2.inpaint(
            frames_pyr[-1], masks_pyr[-1], 7, cv2.INPAINT_NS
        )

        # 逐层上采样并精修
        for i in range(scales - 1, -1, -1):
            h_target, w_target = frames_pyr[i].shape[:2]
            result_up = cv2.pyrUp(result)
            # 裁剪到目标尺寸
            result_up = result_up[:h_target, :w_target]

            # 在当前层用较小半径精修边缘
            mask_current = masks_pyr[i]
            # 只修复掩码边缘区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_core = cv2.erode(mask_current, kernel, iterations=2)
            edge_mask = cv2.subtract(mask_current, mask_core)

            if cv2.countNonZero(edge_mask) > 0:
                result_up = cv2.inpaint(result_up, edge_mask, 3, cv2.INPAINT_NS)

            # 融合: 在掩码区域用修复结果，非掩码区域保留原图
            mask_3ch = cv2.merge([mask_current, mask_current, mask_current])
            result = np.where(mask_3ch > 0, result_up, frames_pyr[i])

        return result

    def _edge_guided_refine(
        self, result: np.ndarray, original: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """边缘引导精修 - 检测原图边缘并引导修复区域的边缘延伸"""
        # 从原图提取边缘
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_orig, 30, 100)

        # 扩展掩码到边缘区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=3)

        # 只取掩码内的边缘 (这些是需要恢复的边缘)
        edges_in_mask = cv2.bitwise_and(edges_dilated, mask)

        if cv2.countNonZero(edges_in_mask) == 0:
            return result

        # 用边缘信息增强修复结果
        # 创建边缘引导权重
        edge_weight = edges_in_mask.astype(np.float32) / 255.0
        edge_weight = cv2.GaussianBlur(edge_weight, (5, 5), 1.0)

        # 沿边缘方向从外部采样颜色
        for angle_step in range(0, 360, 45):
            dx = int(2 * np.cos(np.radians(angle_step)))
            dy = int(2 * np.sin(np.radians(angle_step)))
            shifted = np.roll(np.roll(original, dy, axis=0), dx, axis=1)
            edge_color = cv2.bitwise_and(
                shifted,
                cv2.merge([edges_in_mask, edges_in_mask, edges_in_mask]),
            )
            edge_weight_3ch = cv2.merge([edge_weight] * 3)
            blend = (result.astype(np.float32) * (1 - edge_weight_3ch * 0.3) +
                     edge_color.astype(np.float32) * edge_weight_3ch * 0.3)
            result = np.clip(blend, 0, 255).astype(np.uint8)

        return result

    def _texture_synthesis(
        self, result: np.ndarray, original: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """基于 PatchMatch 的纹理合成 - 从外部区域采样纹理填充"""
        h, w = result.shape[:2]
        patch_size = 7

        # 创建已知区域标记
        known = (mask == 0).astype(np.uint8)

        # 获取掩码边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result

        # 找到所有需要填充的像素 (按距离边界由近到远排序)
        dist_transform = cv2.distanceTransform(
            cv2.bitwise_not(mask), cv2.DIST_L2, 5
        )

        # 分层修复 - 从边界向内
        fill_order = np.argsort(dist_transform[mask > 0])

        all_mask_points = np.argwhere(mask > 0)  # (N, 2) -> (y, x)
        if len(all_mask_points) == 0:
            return result

        sorted_points = all_mask_points[fill_order]

        # 每层取一批像素进行纹理采样
        batch_size = max(100, len(sorted_points) // 20)
        result_work = result.copy()

        for batch_start in range(0, len(sorted_points), batch_size):
            batch_end = min(batch_start + batch_size, len(sorted_points))
            batch = sorted_points[batch_start:batch_end]

            for py, px in batch:
                half = patch_size // 2
                # 边界检查
                y1 = max(0, py - half)
                y2 = min(h, py + half + 1)
                x1 = max(0, px - half)
                x2 = min(w, px + half + 1)

                if y1 >= h or x1 >= w:
                    continue

                # 在已知区域搜索最佳匹配 patch
                best_dist = float("inf")
                best_color = result_work[py, px].astype(np.float32)

                # 搜索范围限制
                search_range = 30
                for _ in range(10):  # 随机采样10次
                    sy = np.random.randint(max(0, py - search_range), min(h, py + search_range))
                    sx = np.random.randint(max(0, px - search_range), min(w, px + search_range))

                    if known[sy, sx] == 0:
                        continue

                    # 比较 patch
                    patch_result = result_work[y1:y2, x1:x2]
                    patch_orig = original[y1:y2, x1:x2]

                    if patch_result.shape != patch_orig.shape:
                        continue

                    # 已知像素的差异
                    known_in_patch = known[y1:y2, x1:x2]
                    diff = np.abs(
                        patch_result.astype(np.float32) - patch_orig.astype(np.float32)
                    )
                    diff = diff * known_in_patch[:, :, np.newaxis]

                    total_known = np.sum(known_in_patch)
                    if total_known > 0:
                        dist = np.sum(diff) / total_known
                        if dist < best_dist:
                            best_dist = dist
                            best_color = original[sy, sx].astype(np.float32)

                # 用找到的最佳颜色替换
                result_work[py, px] = np.clip(best_color, 0, 255).astype(np.uint8)

            # 更新已知区域
            known = (cv2.cvtColor(result_work, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            known = cv2.bitwise_and(known, cv2.bitwise_not(mask_bin))

        return result_work

    def _poisson_blend(
        self, result: np.ndarray, original: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """泊松融合 - 无缝克隆使修复区域与周围自然过渡"""
        try:
            # 找到掩码的中心点
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return result

            # 使用最大轮廓
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            if moments["m00"] == 0:
                return result

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # 确保中心点在图像范围内
            h, w = result.shape[:2]
            cx = max(1, min(cx, w - 2))
            cy = max(1, min(cy, h - 2))

            # 泊松无缝克隆
            blended = cv2.seamlessClone(
                result, original, mask, (cx, cy), cv2.NORMAL_CLONE
            )
            return blended

        except cv2.error:
            # 泊松融合可能在某些情况下失败
            return result

    def _frequency_blend(
        self, result: np.ndarray, original: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """频域融合 - 保持修复区域的频率特性与原图一致"""
        h, w = result.shape[:2]

        # 对每个通道分别处理
        result_float = result.astype(np.float32)
        original_float = original.astype(np.float32)

        # 创建软掩码 (边缘模糊)
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 7)
        soft_mask = soft_mask / 255.0

        # 频域处理
        for c in range(3):
            # FFT
            f_result = np.fft.fft2(result_float[:, :, c])
            f_original = np.fft.fft2(original_float[:, :, c])

            # 获取幅度谱和相位谱
            mag_result = np.abs(f_result)
            phase_result = np.angle(f_result)
            mag_original = np.abs(f_original)
            phase_original = np.angle(f_original)

            # 低频用原图的幅度，保持修复区域的频率分布自然
            # 创建频率权重
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            freq_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            max_freq = np.sqrt(cx**2 + cy**2)
            low_freq_weight = np.clip(1.0 - freq_dist / (max_freq * 0.3), 0, 1)

            # 混合幅度谱
            mag_blended = mag_result * (1 - low_freq_weight * 0.3) + \
                         mag_original * low_freq_weight * 0.3

            # 重构
            f_blended = mag_blended * np.exp(1j * phase_result)
            blended = np.real(np.fft.ifft2(f_blended))

            # 用软掩码混合
            result_float[:, :, c] = (
                result_float[:, :, c] * (1 - soft_mask * 0.15) +
                blended * soft_mask * 0.15
            )

        return np.clip(result_float, 0, 255).astype(np.uint8)

    def _guided_filter_smooth(
        self, result: np.ndarray, original: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """导向滤波 - 使用原图作为引导，保持边缘的平滑"""
        # 使用 OpenCV 的引导滤波 (如果可用)
        try:
            guide = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            # 对每个通道做导向滤波
            smoothed = np.zeros_like(result)
            for c in range(3):
                smoothed[:, :, c] = cv2.ximgproc.guidedFilter(
                    guide=guide,
                    src=result[:, :, c],
                    radius=8,
                    eps=0.02,
                )
            return np.clip(smoothed, 0, 255).astype(np.uint8)
        except (AttributeError, cv2.error):
            # 如果 ximgproc 不可用，使用双边滤波替代
            # 只在掩码区域做平滑
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            expand_mask = cv2.dilate(mask, kernel, iterations=3)

            smoothed = cv2.bilateralFilter(result, 9, 75, 75)
            mask_3ch = cv2.merge([expand_mask, expand_mask, expand_mask])
            return np.where(mask_3ch > 0, smoothed, result)
