"""Data models for scene classification and camera assignment."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

SceneType = Literal["full_court", "closeup"]
Frame = NDArray[np.uint8]


class SceneMlpError(RuntimeError):
    """Raised when the scene-type MLP is missing or cannot classify a frame."""
