from __future__ import annotations

import pandas as pd

from src.camera_assignemnt.embedding_cluster.dataset import _filter_rows


def test_filter_rows_camera_role():
    df = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 1, "sample_role": "camera", "frame_path": "a.jpg"},
            {"scene_id": 0, "frame_number": 5, "sample_role": "ocr", "frame_path": "b.jpg"},
            {"scene_id": 1, "frame_number": 2, "sample_role": "camera", "frame_path": "c.jpg"},
        ]
    )
    out = _filter_rows(df, "camera", middle_image_idx=1)
    assert len(out) == 2
    assert set(out["sample_role"]) == {"camera"}
