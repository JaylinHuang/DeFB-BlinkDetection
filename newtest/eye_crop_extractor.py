"""Eye Crop Extractor — 64×64眼部图像提取模块

从视频帧和人眼定位结果中裁剪并resize出64×64的眼部图像。

支持三种提取方法:
  1. landmark    — 直接从landmark计算眼部bbox, crop+resize (匹配_parse_ann_info)
  2. face_first  — 先裁剪人脸→上采样→再提取眼部 (推荐, 眼睛占比更大)
  3. model_pred  — 使用模型预测的eye_bbox → 反归一化 → 裁剪

输出:
  - 64×64 RGB眼部图像 (PNG)
  - 裁剪结果元数据 (JSON)

landmark索引与 src/data/dataset/mpeblink.py _parse_ann_info 保持一致:
  - 98点WFLW: [33,38,46,50,53,60,65,66,67,72,73,74,75]
  - 68点dlib:  [17,21,22,26,36,40,41,45,46,47,29]

Author: DeFB Lite Team
Created: 2026-06-28
Updated: 2026-06-28 — Added face_first two-stage extraction for better eye occupancy
"""

import os
import json
import time
import sys
from typing import List, Tuple, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


# ============================================================
# 68-Point Facial Landmark Indices (dlib / 300-W format)
# ============================================================
#   Jaw:      0–16
#   Right eyebrow: 17–21
#   Left eyebrow:  22–26
#   Nose bridge: 27–30
#   Nose bottom: 31–35
#   Right eye:     36–41
#   Left eye:      42–47
#   Mouth outer:   48–59
#   Mouth inner:   60–67
#
# 与 mpeblink.py _parse_ann_info 保持一致的68点眼部landmark索引
EYE_REGION_68_IDX = [17, 21, 22, 26, 36, 40, 41, 45, 46, 47, 29]

# 98点WFLW格式
EYE_REGION_98_IDX = [33, 38, 46, 50, 53, 60, 65, 66, 67, 72, 73, 74, 75]

# 简单双眼索引 (用于独立的单眼bbox)
LEFT_EYE_68_IDX = list(range(42, 48))    # person's left eye
RIGHT_EYE_68_IDX = list(range(36, 42))   # person's right eye


def get_eye_region_from_landmarks(
    landmarks: List[List[float]],
    num_landmarks: int = 68,
    padding_ratio: float = 0.1,
    img_width: int = 1280,
    img_height: int = 720,
) -> Tuple[float, float, float, float]:
    """从landmark计算眼部区域bbox (xyxy格式, 像素坐标).

    使用与 mpeblink._parse_ann_info 相同的算法。

    Args:
        landmarks: [[x,y], ...] landmark坐标 (像素坐标系)
        num_landmarks: 68 或 98
        padding_ratio: bbox扩展比例
        img_width, img_height: 图像尺寸

    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    if num_landmarks == 98:
        indices = EYE_REGION_98_IDX
    else:
        indices = EYE_REGION_68_IDX

    # 提取选定landmark的x, y坐标
    pts = []
    for i in indices:
        if i < len(landmarks) and landmarks[i] is not None:
            pts.append(landmarks[i])

    if len(pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 异常值检测
    if max(min_x, max_x, min_y, max_y) > 1000 or min(min_x, max_x, min_y, max_y) < -1000:
        return (0.0, 0.0, 0.0, 0.0)

    w, h = max_x - min_x, max_y - min_y
    x1 = max(min_x - padding_ratio * w, 0)
    y1 = max(min_y - padding_ratio * h, 0)
    x2 = min(max_x + padding_ratio * w, img_width)
    y2 = min(max_y + padding_ratio * h, img_height)

    return (x1, y1, x2, y2)


def face_bbox_to_eye_region(
    face_bbox_xyxy: List[float],
    img_width: int = 1280,
    img_height: int = 720,
    eye_top: float = 0.15,
    eye_bottom: float = 0.55,
) -> Tuple[float, float, float, float]:
    """从人脸bbox近似估算眼部区域 (landmark不可用时的fallback).

    Args:
        face_bbox_xyxy: [x1, y1, x2, y2] 人脸bbox (像素坐标)
        eye_top: 眼部区域在人脸bbox中的相对顶部位置
        eye_bottom: 眼部区域在人脸bbox中的相对底部位置

    Returns:
        (x1, y1, x2, y2) 眼部区域像素坐标
    """
    x1, y1, x2, y2 = face_bbox_xyxy
    face_h = y2 - y1

    ex1 = x1
    ey1 = y1 + eye_top * face_h
    ex2 = x2
    ey2 = y1 + eye_bottom * face_h

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(img_width, ex2)
    ey2 = min(img_height, ey2)

    return (ex1, ey1, ex2, ey2)


def xyxy_to_cxcywh(bbox_xyxy):
    """[x1,y1,x2,y2] → [cx,cy,w,h]"""
    bbox = np.asarray(bbox_xyxy)
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)


def cxcywh_to_xyxy(bbox_cxcywh):
    """[cx,cy,w,h] → [x1,y1,x2,y2]"""
    bbox = np.asarray(bbox_cxcywh)
    cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def _eye_region_from_face_proportions(
    face_w: int, face_h: int,
    eye_top: float = 0.15, eye_bottom: float = 0.50,
    eye_left: float = 0.0, eye_right: float = 1.0,
) -> Tuple[int, int, int, int]:
    """从人脸比例估算眼部区域 (无landmark时的fallback).

    Args:
        face_w, face_h: 人脸图像宽高 (已上采样)
        eye_top, eye_bottom: 眼部在人脸高度中的相对位置
        eye_left, eye_right: 眼部在人脸宽度中的相对位置

    Returns:
        (x1, y1, x2, y2) 眼部区域像素坐标
    """
    ex1 = int(face_w * eye_left)
    ey1 = int(face_h * eye_top)
    ex2 = int(face_w * eye_right)
    ey2 = int(face_h * eye_bottom)
    return (ex1, ey1, ex2, ey2)


def _single_eye_from_face_proportions(
    face_w: int, face_h: int, side: str = 'left',
    eye_top: float = 0.15, eye_bottom: float = 0.48,
) -> Tuple[int, int, int, int]:
    """从人脸比例估算单眼区域 (无landmark时的fallback).

    Args:
        face_w, face_h: 上采样后的人脸宽高
        side: 'left' (左半脸, 图像右侧) 或 'right' (右半脸, 图像左侧)
        eye_top, eye_bottom: 眼部在脸部高度中的相对位置

    Returns:
        (x1, y1, x2, y2) 单眼区域坐标
    """
    ey1 = int(face_h * eye_top)
    ey2 = int(face_h * eye_bottom)

    if side == 'left':
        ex1 = int(face_w * 0.50)
        ex2 = int(face_w * 0.95)
    else:
        ex1 = int(face_w * 0.05)
        ex2 = int(face_w * 0.50)

    return (ex1, ey1, ex2, ey2)


class EyeCropExtractor:
    """64×64眼部图像提取器.

    从视频帧中裁剪眼部区域并resize到固定尺寸(默认64×64).

    Usage:
        extractor = EyeCropExtractor(output_size=(64, 64))
        crop = extractor.extract(frame, eye_bbox_xyxy)
        extractor.save(crop, 'output/eye_00001.png')
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (64, 64),
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.output_size = output_size
        self.interpolation = interpolation

    def extract(self, frame: np.ndarray, eye_bbox_xyxy: np.ndarray) -> np.ndarray:
        """从单帧裁剪眼部区域并resize.

        Args:
            frame: [H, W, 3] BGR图像
            eye_bbox_xyxy: [4] [x1, y1, x2, y2] 像素坐标

        Returns:
            [64, 64, 3] 裁剪并resize后的眼部图像
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in eye_bbox_xyxy]

        # Clamp
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        # Make square by expanding shorter side (preserve aspect ratio, no stretch)
        cw, ch = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(cw, ch)
        x1 = max(0, int(cx - side / 2))
        y1 = max(0, int(cy - side / 2))
        x2 = min(W, int(cx + side / 2))
        y2 = min(H, int(cy + side / 2))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((*self.output_size[::-1], 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, self.output_size, interpolation=self.interpolation)

    def extract_face_first(
        self,
        frame: np.ndarray,
        face_bbox_xyxy: np.ndarray,
        landmarks: List[List[float]] = None,
        num_landmarks: int = 68,
        face_ref_height: int = 256,
        eye_top_ratio: float = 0.15,
        eye_bottom_ratio: float = 0.50,
        eye_left_ratio: float = 0.0,
        eye_right_ratio: float = 1.0,
        padding_ratio: float = 0.1,
    ) -> np.ndarray:
        """两阶段眼部提取: 先裁剪人脸→上采样→再提取眼部.

        解决了原始帧中人脸区域小导致眼部细节不足的问题。
        人脸区域先上采样到face_ref_height高度, 再从放大后的人脸上提取眼部。

        Args:
            frame: [H, W, 3] BGR原始帧
            face_bbox_xyxy: [4] [x1,y1,x2,y2] 人脸bbox (像素坐标)
            landmarks: 68点或98点landmark (可选, 用于精确定位眼部)
            num_landmarks: 68或98
            face_ref_height: 人脸参考高度 (上采样目标)
            eye_top_ratio: 眼部在人脸中的相对顶部位置
            eye_bottom_ratio: 眼部在人脸中的相对底部位置
            eye_left_ratio: 眼部在人脸中的相对左侧
            eye_right_ratio: 眼部在人脸中的相对右侧
            padding_ratio: 眼部区域额外padding

        Returns:
            [64, 64, 3] 裁剪并resize后的眼部图像
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in face_bbox_xyxy]

        # Clamp face bbox
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return np.zeros((*self.output_size, 3), dtype=frame.dtype)

        # Upscale face to reference height
        fh, fw = face_crop.shape[:2]
        scale = face_ref_height / fh
        new_w = int(fw * scale)
        face_up = cv2.resize(face_crop, (new_w, face_ref_height),
                            interpolation=cv2.INTER_CUBIC)

        # Compute eye region in upscaled face coordinates
        if landmarks is not None and len(landmarks) >= 48:
            # Use eye contour landmarks for tight bbox
            if num_landmarks == 98:
                left_idx = list(range(60, 68))   # WFLW left eye
                right_idx = list(range(68, 76))  # WFLW right eye
            else:
                left_idx = list(range(42, 48))   # 68-pt left eye
                right_idx = list(range(36, 42))  # 68-pt right eye

            # Collect eye landmarks, transform to face_crop coords
            eye_pts = []
            for idxs in [left_idx, right_idx]:
                for i in idxs:
                    if i < len(landmarks):
                        pt = landmarks[i]
                        # Transform from original frame coords to face_up coords
                        ex = (pt[0] - x1) * scale
                        ey = (pt[1] - y1) * scale
                        eye_pts.append([ex, ey])

            if eye_pts:
                eye_pts = np.array(eye_pts)
                emin_x, emin_y = eye_pts[:, 0].min(), eye_pts[:, 1].min()
                emax_x, emax_y = eye_pts[:, 0].max(), eye_pts[:, 1].max()

                ew, eh = emax_x - emin_x, emax_y - emin_y
                ex1 = max(0, int(emin_x - padding_ratio * ew))
                ey1 = max(0, int(emin_y - padding_ratio * eh))
                ex2 = min(new_w, int(emax_x + padding_ratio * ew))
                ey2 = min(face_ref_height, int(emax_y + padding_ratio * eh))
            else:
                ex1, ey1, ex2, ey2 = _eye_region_from_face_proportions(
                    new_w, face_ref_height, eye_top_ratio, eye_bottom_ratio,
                    eye_left_ratio, eye_right_ratio)
        else:
            # No landmarks: use face proportions
            ex1, ey1, ex2, ey2 = _eye_region_from_face_proportions(
                new_w, face_ref_height, eye_top_ratio, eye_bottom_ratio,
                eye_left_ratio, eye_right_ratio)

        if ex2 <= ex1 or ey2 <= ey1:
            return np.zeros((*self.output_size, 3), dtype=frame.dtype)

        eye_crop = face_up[ey1:ey2, ex1:ex2]
        return cv2.resize(eye_crop, self.output_size, interpolation=self.interpolation)

    def save(self, crop: np.ndarray, path: str) -> str:
        """保存单张裁剪图像."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, crop)
        return path

    def save_grid(
        self,
        crops: List[np.ndarray],
        path: str,
        cols: int = 10,
        spacing: int = 2,
    ) -> str:
        """拼接多张裁剪为网格图."""
        if not crops:
            return path
        N = len(crops)
        rows = int(np.ceil(N / cols))
        h, w = self.output_size
        grid = np.ones(
            (rows * (h + spacing) - spacing,
             cols * (w + spacing) - spacing, 3),
            dtype=crops[0].dtype,
        ) * 128

        for i, crop in enumerate(crops):
            r, c = i // cols, i % cols
            y0 = r * (h + spacing)
            x0 = c * (w + spacing)
            grid[y0:y0+h, x0:x0+w] = crop

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, grid)
        return path

    # ============================================================
    # Single-eye extraction
    # ============================================================

    def extract_single_eye(
        self,
        frame: np.ndarray,
        face_bbox_xyxy: np.ndarray,
        landmarks: List[List[float]],
        side: str = 'left',
        face_ref_height: int = 256,
        padding_ratio: float = 0.2,
    ) -> np.ndarray:
        """提取单只眼睛的64×64裁剪 (两阶段: face crop → upscale → single eye).

        Args:
            frame: [H, W, 3] BGR原始帧
            face_bbox_xyxy: [4] [x1,y1,x2,y2] 人脸bbox (像素坐标)
            landmarks: 68点或98点landmark列表
            side: 'left' (左眼) 或 'right' (右眼)
            face_ref_height: 人脸上采样目标高度
            padding_ratio: 眼部bbox扩展比例

        Returns:
            [64, 64, 3] 单眼裁剪图像

        中文: 从原始帧中提取单只眼睛的64×64裁剪图像。
        采用两阶段方法: 先裁剪人脸→上采样→再提取单眼。
        side='left'为左眼(landmark 42-47), side='right'为右眼(landmark 36-41)。
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in face_bbox_xyxy]

        # Clamp face bbox
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return np.zeros((*self.output_size, 3), dtype=frame.dtype)

        # Upscale face to reference height
        fh, fw = face_crop.shape[:2]
        scale = face_ref_height / fh
        new_w = int(fw * scale)
        face_up = cv2.resize(face_crop, (new_w, face_ref_height),
                            interpolation=cv2.INTER_CUBIC)

        # Determine single-eye landmark indices
        num_lm = len(landmarks) if landmarks else 0

        if num_lm >= 48:
            if num_lm == 98:
                # WFLW 98-point format
                eye_idx = list(range(60, 68)) if side == 'left' else list(range(68, 76))
            else:
                # 68-point dlib format
                # Left eye (person's left): indices 42-47
                # Right eye (person's right): indices 36-41
                eye_idx = list(range(42, 48)) if side == 'left' else list(range(36, 42))

            # Transform eye landmarks to upscaled face coordinates
            eye_pts = []
            for i in eye_idx:
                if i < num_lm and landmarks[i] is not None:
                    pt = landmarks[i]
                    ex = (pt[0] - x1) * scale
                    ey = (pt[1] - y1) * scale
                    eye_pts.append([ex, ey])

            if eye_pts:
                eye_pts = np.array(eye_pts)
                emin_x, emin_y = eye_pts[:, 0].min(), eye_pts[:, 1].min()
                emax_x, emax_y = eye_pts[:, 0].max(), eye_pts[:, 1].max()

                ew, eh = emax_x - emin_x, emax_y - emin_y
                ex1 = max(0, int(emin_x - padding_ratio * ew))
                ey1 = max(0, int(emin_y - padding_ratio * eh))
                ex2 = min(new_w, int(emax_x + padding_ratio * ew))
                ey2 = min(face_ref_height, int(emax_y + padding_ratio * eh))
            else:
                # Fallback: use face proportions for single eye
                ex1, ey1, ex2, ey2 = _single_eye_from_face_proportions(
                    new_w, face_ref_height, side)
        else:
            # No landmarks: estimate from face proportions
            ex1, ey1, ex2, ey2 = _single_eye_from_face_proportions(
                new_w, face_ref_height, side)

        if ex2 <= ex1 or ey2 <= ey1:
            return np.zeros((*self.output_size, 3), dtype=frame.dtype)

        eye_crop = face_up[ey1:ey2, ex1:ex2]
        return cv2.resize(eye_crop, self.output_size, interpolation=self.interpolation)

    @staticmethod
    def score_sharpness(crop: np.ndarray) -> float:
        """计算裁剪图像的清晰度分数 (Laplacian variance).

        值越高 = 边缘越清晰 = 眼部细节越好。
        中文: 使用Laplacian方差衡量图像清晰度, 分数越高眼睛越清晰。

        Args:
            crop: [H, W, 3] BGR图像

        Returns:
            float: 清晰度分数
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def find_best_window(
        self,
        frames: List[np.ndarray],
        face_bboxes: List[List[float]],
        landmarks: List[List[List[float]]],
        side: str = 'left',
        window_size: int = 13,
        face_ref_height: int = 256,
    ) -> Tuple[int, np.ndarray, float]:
        """在视频帧序列中查找最佳的连续N帧单眼窗口.

        使用滑动窗口扫描所有可能的连续window_size帧序列,
        对每个窗口计算平均清晰度分数, 返回得分最高的窗口。

        中文: 滑动窗口扫描所有13帧连续序列, 选取眼部最清晰的那个窗口。

        Args:
            frames: 帧图像列表 [frame_0, frame_1, ...]
            face_bboxes: 每帧的人脸bbox [x,y,w,h] 列表
            landmarks: 每帧的landmark列表
            side: 'left' 或 'right'
            window_size: 窗口大小 (默认13)
            face_ref_height: 人脸上采样高度

        Returns:
            (start_idx, crops_window, avg_score)
            - start_idx: 最佳窗口的起始帧索引
            - crops_window: [window_size, 64, 64, 3] numpy数组
            - avg_score: 窗口平均清晰度分数
        """
        n_frames = len(frames)

        if n_frames < window_size:
            # 帧数不足: 用所有可用帧, 复制首尾帧填充
            crops = []
            for i in range(n_frames):
                fb = face_bboxes[i]
                if fb is None:
                    continue
                face_xyxy = np.array([fb[0], fb[1], fb[0]+fb[2], fb[1]+fb[3]])
                crop = self.extract_single_eye(
                    frames[i], face_xyxy, landmarks[i] if landmarks else None,
                    side, face_ref_height)
                crops.append(crop)

            if len(crops) == 0:
                empty = np.zeros((window_size, *self.output_size, 3), dtype=np.uint8)
                return (0, empty, 0.0)

            # Pad to window_size
            while len(crops) < window_size:
                crops.insert(0, crops[0])
                crops.append(crops[-1])
            crops = crops[:window_size]

            result = np.stack(crops, axis=0)
            scores = [self.score_sharpness(c) for c in crops]
            return (0, result, np.mean(scores) if scores else 0.0)

        best_start = 0
        best_score = -1.0
        best_crops = None

        for start in range(n_frames - window_size + 1):
            window_crops = []
            window_scores = []

            for offset in range(window_size):
                idx = start + offset
                fb = face_bboxes[idx]
                if fb is None:
                    break
                face_xyxy = np.array([fb[0], fb[1], fb[0]+fb[2], fb[1]+fb[3]])
                lm = landmarks[idx] if landmarks and idx < len(landmarks) else None
                crop = self.extract_single_eye(
                    frames[idx], face_xyxy, lm, side, face_ref_height)
                window_crops.append(crop)
                window_scores.append(self.score_sharpness(crop))

            if len(window_crops) < window_size:
                continue

            avg_score = np.mean(window_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_start = start
                best_crops = np.stack(window_crops, axis=0)

        if best_crops is None:
            # Fallback: use first window
            best_start = 0
            best_crops = np.zeros((window_size, *self.output_size, 3), dtype=np.uint8)
            best_score = 0.0

        return (best_start, best_crops, float(best_score))




def extract_eye_crops_gt(
    ann_file: str,
    rawframes_dir: str,
    output_dir: str,
    max_persons: int = 10,
    max_frames_per_person: int = 30,
    method: str = 'face_first',
) -> dict:
    """从GT landmark提取眼部裁剪 (无需模型).

    Args:
        ann_file: 标注JSON文件路径
        rawframes_dir: 原始帧目录
        output_dir: 输出根目录
        max_persons: 最多处理人数
        max_frames_per_person: 每人最多处理帧数
        method: 'face_first' (推荐, 两阶段, 眼睛占比大) 或 'landmark' (直接裁剪)

    Returns:
        {'total_crops': int, 'results': list, 'output_dir': str}
    """
    print("=" * 60)
    print(f"Eye Crop Extraction — GT (method={method})")
    print("=" * 60)

    extractor = EyeCropExtractor()
    os.makedirs(output_dir, exist_ok=True)

    with open(ann_file, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    print(f"  Annotations: {len(annotations)} persons")
    print(f"  Method:      {method}")
    print(f"  Output dir:  {output_dir}")

    all_results = []
    total_crops = 0

    for anno in annotations[:max_persons]:
        video_id = anno.get('video_id', 0)
        landmarks = anno.get('landmark', [])
        face_bboxes = anno.get('bboxes', [])

        if not landmarks:
            continue

        video_dir = os.path.join(rawframes_dir, str(video_id))
        if not os.path.isdir(video_dir):
            print(f"  [SKIP] Video {video_id}: rawframes dir not found")
            continue

        frame_files = sorted(os.listdir(video_dir))
        if not frame_files:
            continue

        n_frames = min(len(frame_files), len(landmarks), max_frames_per_person)
        person_id = anno.get('id', 0)

        # 确定landmark格式
        first_lm = landmarks[0] if landmarks else None
        num_lm = len(first_lm) if first_lm and isinstance(first_lm, list) else 68

        person_crops = []
        person_dir = os.path.join(output_dir, str(video_id), f"person_{person_id:04d}")
        os.makedirs(person_dir, exist_ok=True)

        for i in range(n_frames):
            lm = landmarks[i]
            face_bbox = face_bboxes[i] if i < len(face_bboxes) else None

            if not lm:
                continue

            frame_path = os.path.join(video_dir, frame_files[i])
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            if method == 'face_first' and face_bbox:
                # 两阶段: 人脸→上采样→眼部
                face_xyxy = [face_bbox[0], face_bbox[1],
                             face_bbox[0] + face_bbox[2],
                             face_bbox[1] + face_bbox[3]]
                crop = extractor.extract_face_first(
                    frame, np.array(face_xyxy), lm, num_lm)
            else:
                # 直接法: landmark→眼部bbox→裁剪
                eye_xyxy = get_eye_region_from_landmarks(lm, num_lm)
                if eye_xyxy[2] <= eye_xyxy[0] or eye_xyxy[3] <= eye_xyxy[1]:
                    if face_bbox:
                        eye_xyxy = face_bbox_to_eye_region(face_bbox)
                    else:
                        continue
                crop = extractor.extract(frame, np.array(eye_xyxy))

            crop_path = os.path.join(person_dir, f"frame_{i:05d}.png")
            extractor.save(crop, crop_path)
            person_crops.append(crop)
            total_crops += 1

            all_results.append({
                'video_id': video_id,
                'person_id': person_id,
                'frame_idx': i,
                'crop_path': crop_path,
            })

        if person_crops:
            grid_path = os.path.join(person_dir, 'grid.png')
            extractor.save_grid(person_crops, grid_path)
            print(f"  Video {video_id} Person {person_id}: "
                  f"{len(person_crops)} crops → {person_dir}")

    # 保存元数据
    meta = {
        'total_crops': total_crops,
        'output_size': [64, 64],
        'method': method,
        'results': all_results,
    }
    meta_path = os.path.join(output_dir, 'extraction_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Total: {total_crops} crops (64×64)")
    print(f"  Meta:  {meta_path}")
    return meta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='64×64眼部图像提取 (GT Landmark)')
    parser.add_argument('--ann', type=str,
                        default='E:/documents/mssb code/mpeblink2.0/mpeblink_lite/annotations/train_mini.json')
    parser.add_argument('--rawframes', type=str,
                        default='E:/documents/mssb code/mpeblink2.0/mpeblink_lite/train_rawframes')
    parser.add_argument('--output', type=str, default='./output/eye_crops_gt')
    parser.add_argument('--max-persons', type=int, default=10)
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--method', type=str, default='face_first',
                        choices=['face_first', 'landmark'],
                        help='face_first: 人脸→上采样→眼部 (推荐); landmark: 直接裁剪')
    args = parser.parse_args()

    extract_eye_crops_gt(
        args.ann, args.rawframes, args.output,
        max_persons=args.max_persons,
        max_frames_per_person=args.max_frames,
        method=args.method,
    )
