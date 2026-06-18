from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.scene_extractor import load_scenes
from broadcast_pipeline.viz.camera_compare import parse_vote_counts

SlotName = Literal["begin", "mid", "end"]


class CameraCollageLoadError(Exception):
    """Raised when camera collage artifacts are missing or invalid."""


@dataclass
class SceneFrameSlot:
    slot: SlotName
    frame_number: int
    frame_path: str


@dataclass
class SceneCollageEntry:
    scene_id: int
    camera_id: str
    start_sec: float | None
    end_sec: float | None
    frames: list[SceneFrameSlot]
    cluster_id: int = -1
    camera_vote_counts: dict[str, int] | None = None
    unanimous: bool = True
    winner_share: float = 1.0
    pred_noise: bool = False
    gt_camera: str | None = None


@dataclass
class CameraCollageBundle:
    output_dir: Path
    camera_ids: list[str]
    scenes_by_camera: dict[str, list[SceneCollageEntry]]
    slot_lookup: dict[tuple[int, SlotName], SceneFrameSlot]
    has_debug_artifact: bool = False
    camera_cluster_ids: dict[str, int] | None = None

    def scene_count(self, camera_id: str) -> int:
        return len(self.scenes_by_camera.get(camera_id, []))

    def scenes_for_camera(self, camera_id: str) -> list[SceneCollageEntry]:
        return list(self.scenes_by_camera.get(camera_id, []))

    def resolve_slot_path(self, scene_id: int, slot: SlotName) -> Path | None:
        info = self.slot_lookup.get((scene_id, slot))
        if info is None:
            return None
        return Path(info.frame_path)


def pick_scene_slots(
    frame_rows: list[dict],
    output_dir: Path,
) -> list[SceneFrameSlot]:
    """Pick begin, mid, and end camera sample frames for one scene."""
    if not frame_rows:
        return []

    ordered = sorted(frame_rows, key=lambda row: int(row["frame_number"]))
    indices = [0, len(ordered) // 2, len(ordered) - 1]
    slot_names: list[SlotName] = ["begin", "mid", "end"]

    seen_frames: set[int] = set()
    slots: list[SceneFrameSlot] = []
    for idx, name in zip(indices, slot_names, strict=True):
        row = ordered[idx]
        frame_number = int(row["frame_number"])
        if frame_number in seen_frames:
            continue
        seen_frames.add(frame_number)
        path = Path(str(row["frame_path"]))
        if not path.is_absolute():
            path = (output_dir / path).resolve()
        slots.append(
            SceneFrameSlot(
                slot=name,
                frame_number=frame_number,
                frame_path=str(path),
            )
        )
    return slots


def _sort_camera_ids(ids: list[str]) -> list[str]:
    def key(camera_id: str) -> tuple[int, str]:
        if camera_id == "unknown":
            return (2, camera_id)
        if camera_id.startswith("cam_"):
            try:
                return (0, f"{int(camera_id.split('_', 1)[1]):08d}")
            except ValueError:
                pass
        return (1, camera_id)

    return sorted(set(ids), key=key)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_gt_lookup(output_dir: Path) -> dict[int, str]:
    path = output_dir / "camera_assignment_analysis.json"
    if not path.is_file():
        return {}
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    lookup: dict[int, str] = {}
    for row in payload.get("scene_detail", []):
        if row.get("scene_id") is not None and row.get("camera_id") is not None:
            lookup[int(row["scene_id"])] = str(row["camera_id"])
    return lookup


def load_camera_collage_bundle(output_dir: Path) -> CameraCollageBundle:
    config = PipelineConfig(output_dir=Path(output_dir))
    scene_assignments_path = config.artifact("scene_assignments")
    frame_index_path = config.artifact("frame_index")

    if not scene_assignments_path.is_file():
        raise CameraCollageLoadError(
            f"Missing {scene_assignments_path.name} in {config.output_dir}. "
            "Run the pipeline through the cameras stage first."
        )
    if not frame_index_path.is_file():
        raise CameraCollageLoadError(
            f"Missing {frame_index_path.name} in {config.output_dir}. "
            "Run the pipeline through the extract stage first."
        )

    scene_assignments = _read_csv(scene_assignments_path)
    frame_index = _read_csv(frame_index_path)
    if scene_assignments.empty:
        raise CameraCollageLoadError("scene_assignments.csv is empty.")

    scenes_path = config.artifact("scenes")
    scene_timing: dict[int, tuple[float | None, float | None]] = {}
    if scenes_path.is_file():
        for scene in load_scenes(scenes_path):
            scene_timing[scene.scene_id] = (scene.start_sec, scene.end_sec)

    camera_frames = frame_index[frame_index["sample_role"] == "camera"].copy()
    frames_by_scene: dict[int, list[dict]] = {}
    for row in camera_frames.itertuples(index=False):
        scene_id = int(row.scene_id)
        frames_by_scene.setdefault(scene_id, []).append(
            {
                "frame_number": int(row.frame_number),
                "frame_path": str(row.frame_path),
            }
        )

    scenes_by_camera: dict[str, list[SceneCollageEntry]] = {}
    slot_lookup: dict[tuple[int, SlotName], SceneFrameSlot] = {}
    gt_lookup = _load_gt_lookup(config.output_dir)

    for row in scene_assignments.itertuples(index=False):
        scene_id = int(row.scene_id)
        camera_id = str(getattr(row, "camera_id", "unknown"))
        cluster_id = int(getattr(row, "cluster_id", -1))
        votes = parse_vote_counts(getattr(row, "camera_vote_counts_json", ""))
        total_votes = sum(votes.values())
        winner_share = float(votes.get(camera_id, 0) / total_votes) if total_votes else 1.0
        start_sec, end_sec = scene_timing.get(scene_id, (None, None))
        slots = pick_scene_slots(frames_by_scene.get(scene_id, []), config.output_dir)
        entry = SceneCollageEntry(
            scene_id=scene_id,
            camera_id=camera_id,
            start_sec=start_sec,
            end_sec=end_sec,
            frames=slots,
            cluster_id=cluster_id,
            camera_vote_counts=votes,
            unanimous=len(votes) <= 1,
            winner_share=winner_share,
            pred_noise=cluster_id < 0,
            gt_camera=gt_lookup.get(scene_id),
        )
        scenes_by_camera.setdefault(camera_id, []).append(entry)
        for slot in slots:
            slot_lookup[(scene_id, slot.slot)] = slot

    for entries in scenes_by_camera.values():
        entries.sort(key=lambda item: item.scene_id)

    camera_ids = _sort_camera_ids(scene_assignments["camera_id"].astype(str).tolist())
    camera_cluster_ids: dict[str, int] = {}
    for camera_id in camera_ids:
        clusters = scene_assignments[scene_assignments["camera_id"].astype(str) == camera_id][
            "cluster_id"
        ]
        if not clusters.empty:
            camera_cluster_ids[camera_id] = int(clusters.mode().iloc[0])

    has_debug = config.artifact("camera_clustering_debug").is_file()

    return CameraCollageBundle(
        output_dir=config.output_dir,
        camera_ids=camera_ids,
        scenes_by_camera=scenes_by_camera,
        slot_lookup=slot_lookup,
        has_debug_artifact=has_debug,
        camera_cluster_ids=camera_cluster_ids,
    )


def scene_entry_to_dict(entry: SceneCollageEntry) -> dict:
    return {
        "scene_id": entry.scene_id,
        "camera_id": entry.camera_id,
        "start_sec": entry.start_sec,
        "end_sec": entry.end_sec,
        "cluster_id": entry.cluster_id,
        "camera_vote_counts": entry.camera_vote_counts or {},
        "unanimous": entry.unanimous,
        "winner_share": entry.winner_share,
        "pred_noise": entry.pred_noise,
        "gt_camera": entry.gt_camera,
        "frames": [
            {
                "slot": slot.slot,
                "frame_number": slot.frame_number,
                "image_url": f"/api/scene-images/{entry.scene_id}/{slot.slot}",
            }
            for slot in entry.frames
        ],
    }
