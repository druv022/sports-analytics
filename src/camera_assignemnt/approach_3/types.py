from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KeypointDetection:
    points: np.ndarray
    valid: np.ndarray
    raw_points: np.ndarray
