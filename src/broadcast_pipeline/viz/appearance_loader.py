from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from broadcast_pipeline.appearance_compat import (
    appearance_config_from_pipeline,
    is_appearance_eligible,
    load_scene_appearances,
)
from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.viz.camera_collage import pick_scene_slots
from src.person_appearance.signature import build_compatibility_components, normalize_signature
from src.person_appearance.types import SceneAppearance


class AppearanceLoadError(Exception):
    """Raised when appearance artifacts are missing or invalid."""


@dataclass(frozen=True)
class FrameAppearanceRow:
    scene_id: int
    frame_number: int
    frame_path: str
    person_count: int
    person_colors: tuple[str, ...]
    confidence: float
    status: str
    primary_bgr: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class SceneAppearanceRow:
    scene_id: int
    scene_type: str
    person_count: int
    person_colors: tuple[str, ...]
    appearance_signature: str
    confidence: float
    status: str
    primary_bgr: tuple[int, int, int] | None = None
    dominant_track_frames: int = 0
    dominant_track_median_area: int = 0
    camera_id: str | None = None
    has_count_variance: bool = False
    compatibility_component: int | None = None


@dataclass
class AppearanceBundle:
    output_dir: Path
    has_appearance_artifacts: bool
    scene_by_id: dict[int, SceneAppearanceRow] = field(default_factory=dict)
    frames_by_scene: dict[int, list[FrameAppearanceRow]] = field(default_factory=dict)
    frames_by_number: dict[int, FrameAppearanceRow] = field(default_factory=dict)
    _scene_appearances: list[SceneAppearance] = field(default_factory=list)
    _config: PipelineConfig | None = None

    def build_summary(self) -> dict:
        scenes = list(self.scene_by_id.values())
        frames = [
            frame
            for frame_list in self.frames_by_scene.values()
            for frame in frame_list
        ]
        count_hist: Counter[int] = Counter()
        status_counts: Counter[str] = Counter()
        scene_type_counts: Counter[str] = Counter()

        for scene in scenes:
            count_hist[scene.person_count] += 1
            status_counts[scene.status] += 1
            scene_type_counts[scene.scene_type] += 1

        frame_status_counts: Counter[str] = Counter(frame.status for frame in frames)
        variance_scenes = sum(1 for scene in scenes if scene.has_count_variance)
        issue_scene_ids = self.scenes_with_issues()

        groups = self.compatibility_groups()
        return {
            "n_scenes": len(scenes),
            "n_frames": len(frames),
            "person_count_histogram": dict(sorted(count_hist.items())),
            "scene_status_counts": dict(status_counts),
            "frame_status_counts": dict(frame_status_counts),
            "scene_type_counts": dict(scene_type_counts),
            "n_scenes_with_count_variance": variance_scenes,
            "n_scenes_with_issues": len(issue_scene_ids),
            "n_compatibility_components": len({cid for cid in groups.values()}),
        }

    def scene_frame_variance(self, scene_id: int) -> bool:
        scene = self.scene_by_id.get(scene_id)
        if scene is None:
            return False
        frames = self.frames_by_scene.get(scene_id, [])
        if not frames:
            return False
        return any(frame.person_count != scene.person_count for frame in frames)

    def compatibility_groups(self) -> dict[int, int]:
        if not self._scene_appearances or self._config is None:
            return {}
        eligible_ids = {
            app.scene_id
            for app in self._scene_appearances
            if is_appearance_eligible(app)
        }
        if not eligible_ids:
            return {}
        appearance_cfg = appearance_config_from_pipeline(self._config)
        return build_compatibility_components(
            self._scene_appearances,
            appearance_cfg,
            eligible_scene_ids=eligible_ids,
        )

    def scenes_with_issues(self) -> list[int]:
        issues: list[int] = []
        for scene_id, scene in self.scene_by_id.items():
            if scene.has_count_variance:
                issues.append(scene_id)
                continue
            if scene.scene_type == "closeup" and scene.status in {"low_conf", "no_person"}:
                issues.append(scene_id)
        return sorted(set(issues))

    def scene_list(self) -> list[SceneAppearanceRow]:
        return sorted(self.scene_by_id.values(), key=lambda row: row.scene_id)

    def scene_detail(self, scene_id: int) -> dict | None:
        scene = self.scene_by_id.get(scene_id)
        if scene is None:
            return None
        frames = sorted(
            self.frames_by_scene.get(scene_id, []),
            key=lambda row: row.frame_number,
        )
        slot_rows: list[dict] = []
        frame_dicts = [
            {"frame_number": int(row["frame_number"]), "frame_path": str(row["frame_path"])}
            for row in (
                {
                    "frame_number": frame.frame_number,
                    "frame_path": frame.frame_path,
                }
                for frame in frames
            )
        ]
        for slot in pick_scene_slots(frame_dicts, self.output_dir):
            slot_rows.append(
                {
                    "slot": slot.slot,
                    "frame_number": slot.frame_number,
                    "image_url": f"/api/scene-images/{scene_id}/{slot.slot}",
                }
            )

        return {
            "scene": _scene_row_to_dict(scene),
            "frames": [_frame_row_to_dict(frame) for frame in frames],
            "slots": slot_rows,
        }


def _parse_colors(raw: object) -> tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    return tuple(str(c) for c in parsed)


def _parse_bgr(raw: object) -> tuple[int, int, int] | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        parsed = json.loads(raw) if raw else []
    else:
        parsed = raw
    if not parsed or len(parsed) < 3:
        return None
    return (int(parsed[0]), int(parsed[1]), int(parsed[2]))


def scene_row_to_dict(row: SceneAppearanceRow) -> dict:
    return _scene_row_to_dict(row)


def frame_row_to_dict(row: FrameAppearanceRow) -> dict:
    return _frame_row_to_dict(row)


def _frame_row_to_dict(row: FrameAppearanceRow) -> dict:
    return {
        "scene_id": row.scene_id,
        "frame_number": row.frame_number,
        "frame_path": row.frame_path,
        "person_count": row.person_count,
        "person_colors": list(row.person_colors),
        "primary_bgr": list(row.primary_bgr) if row.primary_bgr else None,
        "confidence": row.confidence,
        "status": row.status,
    }


def _scene_row_to_dict(row: SceneAppearanceRow) -> dict:
    return {
        "scene_id": row.scene_id,
        "scene_type": row.scene_type,
        "person_count": row.person_count,
        "person_colors": list(row.person_colors),
        "appearance_signature": row.appearance_signature,
        "primary_bgr": list(row.primary_bgr) if row.primary_bgr else None,
        "dominant_track_frames": row.dominant_track_frames,
        "dominant_track_median_area": row.dominant_track_median_area,
        "confidence": row.confidence,
        "status": row.status,
        "camera_id": row.camera_id,
        "has_count_variance": row.has_count_variance,
        "compatibility_component": row.compatibility_component,
    }


def _load_camera_lookup(output_dir: Path) -> dict[int, str]:
    path = output_dir / "scene_assignments.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    lookup: dict[int, str] = {}
    for row in df.itertuples(index=False):
        if getattr(row, "scene_id", None) is not None:
            lookup[int(row.scene_id)] = str(getattr(row, "camera_id", "unknown"))
    return lookup


def load_appearance_bundle(output_dir: Path) -> AppearanceBundle:
    config = PipelineConfig(output_dir=Path(output_dir))
    frame_path = config.artifact("frame_appearance")
    scene_path = config.artifact("scene_appearance")

    has_artifacts = frame_path.is_file() and scene_path.is_file()
    if not has_artifacts:
        return AppearanceBundle(
            output_dir=config.output_dir,
            has_appearance_artifacts=False,
            _config=config,
        )

    frame_df = pd.read_csv(frame_path)
    scene_df = pd.read_csv(scene_path)
    camera_lookup = _load_camera_lookup(config.output_dir)

    frames_by_scene: dict[int, list[FrameAppearanceRow]] = {}
    frames_by_number: dict[int, FrameAppearanceRow] = {}

    for row in frame_df.itertuples(index=False):
        frame_row = FrameAppearanceRow(
            scene_id=int(row.scene_id),
            frame_number=int(row.frame_number),
            frame_path=str(row.frame_path),
            person_count=int(row.person_count),
            person_colors=_parse_colors(getattr(row, "person_colors_json", "[]")),
            confidence=float(getattr(row, "confidence", 0.0)),
            status=str(getattr(row, "status", "error")),
            primary_bgr=_parse_bgr(getattr(row, "primary_bgr_json", None)),
        )
        frames_by_scene.setdefault(frame_row.scene_id, []).append(frame_row)
        frames_by_number[frame_row.frame_number] = frame_row

    scene_appearances = load_scene_appearances(config)
    compatibility = build_compatibility_components(
        scene_appearances,
        appearance_config_from_pipeline(config),
        eligible_scene_ids={
            app.scene_id for app in scene_appearances if is_appearance_eligible(app)
        },
    )

    scene_by_id: dict[int, SceneAppearanceRow] = {}
    for row in scene_df.itertuples(index=False):
        scene_id = int(row.scene_id)
        scene_row = SceneAppearanceRow(
            scene_id=scene_id,
            scene_type=str(getattr(row, "scene_type", "closeup")),
            person_count=int(row.person_count),
            person_colors=_parse_colors(getattr(row, "person_colors_json", "[]")),
            appearance_signature=normalize_signature(str(getattr(row, "appearance_signature", ""))),
            confidence=float(getattr(row, "confidence", 0.0)),
            status=str(getattr(row, "status", "error")),
            primary_bgr=_parse_bgr(getattr(row, "primary_bgr_json", None)),
            dominant_track_frames=int(getattr(row, "dominant_track_frames", 0)),
            dominant_track_median_area=int(getattr(row, "dominant_track_median_area", 0)),
            camera_id=camera_lookup.get(scene_id),
        )
        has_variance = any(
            frame.person_count != scene_row.person_count
            for frame in frames_by_scene.get(scene_id, [])
        )
        scene_by_id[scene_id] = SceneAppearanceRow(
            scene_id=scene_row.scene_id,
            scene_type=scene_row.scene_type,
            person_count=scene_row.person_count,
            person_colors=scene_row.person_colors,
            appearance_signature=scene_row.appearance_signature,
            confidence=scene_row.confidence,
            status=scene_row.status,
            primary_bgr=scene_row.primary_bgr,
            dominant_track_frames=scene_row.dominant_track_frames,
            dominant_track_median_area=scene_row.dominant_track_median_area,
            camera_id=scene_row.camera_id,
            has_count_variance=has_variance,
            compatibility_component=compatibility.get(scene_id),
        )

    return AppearanceBundle(
        output_dir=config.output_dir,
        has_appearance_artifacts=True,
        scene_by_id=scene_by_id,
        frames_by_scene=frames_by_scene,
        frames_by_number=frames_by_number,
        _scene_appearances=scene_appearances,
        _config=config,
    )
