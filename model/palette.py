"""固定色板（§11.13 颜色分类方案）。

回归损失下"全员灰"是理性解（均值平衡点）；把 solid fill 颜色离散到
固定色板并改用 CE 分类监督，消灭均值退路。渐变 stop 颜色暂保持连续。
"""
from __future__ import annotations

import colorsys

import numpy as np

# 48 色 = 12 hue × 2 饱和 × 2 明度
N_PALETTE = 48
_HUES = np.arange(12) / 12.0
_SATS = (0.50, 0.95)
_VALS = (0.40, 0.85)

PALETTE_RGB: np.ndarray = None  # [48, 3] float32


def _build() -> np.ndarray:
    cols = []
    for h in _HUES:
        for s in _SATS:
            for v in _VALS:
                r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
                cols.append((r, g, b))
    return np.asarray(cols, dtype=np.float32)


PALETTE_RGB = _build()


def sample_palette_rgb(rng) -> tuple:
    """generator 用：从色板均匀采样一个颜色。"""
    i = int(rng.integers(0, N_PALETTE))
    c = PALETTE_RGB[i]
    return (float(c[0]), float(c[1]), float(c[2]))


def rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    """GT rgb → 最近色板 id（[N,3] → [N]）。"""
    d = np.linalg.norm(rgb[:, None, :] - PALETTE_RGB[None, :, :], axis=2)
    return d.argmin(axis=1)
