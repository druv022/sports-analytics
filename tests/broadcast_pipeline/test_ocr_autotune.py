from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.ocr_autotune import (
    OcrAutotuneOptions,
    OcrAutotuneReport,
    OcrBenchmarkResult,
    autotune_ocr_settings,
    collect_ocr_frame_paths,
    ensure_ocr_prerequisites,
    run_ocr_autotune_pipeline,
)


def test_collect_ocr_frame_paths_from_frame_index(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "scene_0_frame_10.jpg"
    frame_path.write_bytes(b"x")

    output_dir = tmp_path / "pipeline"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "frame_path": str(frame_path),
                "sample_role": "ocr",
            },
            {
                "scene_id": 0,
                "frame_number": 20,
                "frame_path": str(frames_dir / "missing.jpg"),
                "sample_role": "ocr",
            },
        ]
    ).to_csv(output_dir / "frame_index.csv", index=False)

    config = PipelineConfig(output_dir=output_dir)
    paths = collect_ocr_frame_paths(config, limit=5)
    assert paths == [frame_path]


def test_ensure_ocr_prerequisites_runs_missing_stages(tmp_path):
    config = PipelineConfig(output_dir=tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_run(cfg, start, end):
        calls.append((start, end))
        if end == "extract":
            pd.DataFrame(
                [
                    {
                        "scene_id": 0,
                        "frame_number": 10,
                        "frame_path": str(tmp_path / "f.jpg"),
                        "sample_role": "ocr",
                    }
                ]
            ).to_csv(cfg.artifact("frame_index"), index=False)
        if end == "cameras":
            pd.DataFrame(
                [
                    {
                        "scene_id": 0,
                        "frame_number": 10,
                        "camera_id": "cam_0",
                        "sample_role": "ocr",
                    }
                ]
            ).to_csv(cfg.artifact("frame_assignments"), index=False)
        return MagicMock()

    actions = ensure_ocr_prerequisites(
        config,
        fake_run,
        OcrAutotuneOptions(),
    )
    assert actions == ["meta → extract", "filter → cameras"]
    assert calls == [("meta", "extract"), ("filter", "cameras")]


def test_autotune_prefers_prefetch_when_faster(tmp_path):
    config = PipelineConfig(output_dir=tmp_path, accelerator="cpu")
    paths = [tmp_path / f"f{i}.jpg" for i in range(3)]
    for path in paths:
        path.write_bytes(b"x")

    sync = MagicMock(fps=1.0, p50_s=1.0, p95_s=1.0, elapsed_s=3.0, frame_count=3, label="sync")
    prefetch = MagicMock(fps=2.0, p50_s=0.4, p95_s=0.5, elapsed_s=1.5, frame_count=3, label="prefetch")

    with (
        patch("broadcast_pipeline.ocr_autotune.verify_ocr_accelerator", return_value=("cpu", "cpu", [])),
        patch("broadcast_pipeline.ocr_autotune.benchmark_ocr_paths", side_effect=[sync, prefetch]),
        patch("broadcast_pipeline.ocr_autotune.clear_ocr_engine_cache"),
    ):
        report = autotune_ocr_settings(
            config,
            paths,
            OcrAutotuneOptions(),
            locked={},
        )

    assert report.chosen_prefetch_workers == 2
    assert config.ocr_prefetch_workers == 2


def test_autotune_respects_locked_prefetch(tmp_path):
    config = PipelineConfig(output_dir=tmp_path, ocr_prefetch_workers=0, accelerator="cpu")
    paths = [tmp_path / "f.jpg"]
    paths[0].write_bytes(b"x")

    sync = MagicMock(fps=1.0, p50_s=1.0, p95_s=1.0, elapsed_s=1.0, frame_count=1, label="sync")
    prefetch = MagicMock(fps=5.0, p50_s=0.1, p95_s=0.2, elapsed_s=0.2, frame_count=1, label="prefetch")

    with (
        patch("broadcast_pipeline.ocr_autotune.verify_ocr_accelerator", return_value=("cpu", "cpu", [])),
        patch("broadcast_pipeline.ocr_autotune.benchmark_ocr_paths", side_effect=[sync, prefetch]),
        patch("broadcast_pipeline.ocr_autotune.clear_ocr_engine_cache"),
    ):
        report = autotune_ocr_settings(
            config,
            paths,
            OcrAutotuneOptions(),
            locked={"ocr_prefetch_workers": 0},
        )

    assert report.chosen_prefetch_workers == 0
    assert config.ocr_prefetch_workers == 0


def test_run_ocr_autotune_pipeline_invokes_ocr_stage(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"x")
    output_dir = tmp_path / "pipeline"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "frame_path": str(frame_path),
                "sample_role": "ocr",
            }
        ]
    ).to_csv(output_dir / "frame_index.csv", index=False)
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "camera_id": "cam_0",
                "sample_role": "ocr",
            }
        ]
    ).to_csv(output_dir / "frame_assignments.csv", index=False)

    config = PipelineConfig(output_dir=output_dir, accelerator="cpu")
    ocr_calls: list[tuple[str, str]] = []

    def fake_run(cfg, start, end):
        ocr_calls.append((start, end))
        summary = MagicMock()
        summary.n_frames = 1
        return summary

    with (
        patch("broadcast_pipeline.ocr_autotune.autotune_ocr_settings") as mock_tune,
        patch("broadcast_pipeline.ocr_autotune.clear_ocr_engine_cache"),
    ):
        mock_tune.return_value = OcrAutotuneReport(
            torch_device="cpu",
            ocr_backend="cpu",
            sync=OcrBenchmarkResult("sync", 1.0, 1.0, 1.0, 1.0, 1),
            prefetch=OcrBenchmarkResult("prefetch", 2.0, 0.5, 0.6, 0.5, 1),
            chosen_prefetch_workers=2,
            chosen_rec_batch=6,
            chosen_cls_batch=6,
        )
        with patch("broadcast_pipeline.ocr_autotune.print_autotune_report"):
            run_ocr_autotune_pipeline(config, fake_run, run_ocr=True)

    assert ("ocr", "ocr") in ocr_calls
