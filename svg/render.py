from __future__ import annotations

import io

import numpy as np
import resvg_py
from PIL import Image

from .serializer import serialize
from .scene_graph import Scene


def svg_to_rgba(svg_string: str, width: int, height: int) -> np.ndarray:
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg_string, width=width, height=height))
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def render_scene_resvg(scene: Scene, width: int = None, height: int = None) -> np.ndarray:
    w = int(width if width is not None else scene.width)
    h = int(height if height is not None else scene.height)
    return svg_to_rgba(serialize(scene), w, h)
