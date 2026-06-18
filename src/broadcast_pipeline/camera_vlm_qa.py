"""Optional VLM-based consistency check for multi-scene camera collages."""

from __future__ import annotations

import json
from pathlib import Path

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.viz.camera_collage import load_camera_collage_bundle

VLM_COLLAGE_PROMPT = (
    "You are verifying broadcast camera clustering. Each row is a different scene "
    "assigned to the same camera. Do all rows show the same physical camera viewpoint "
    '(same angle, framing, and background)? Return JSON only: '
    '{"consistent": true|false, "outlier_scene_ids": [int, ...], "reason": "..."}'
)


def run_camera_collage_vlm_qa(config: PipelineConfig) -> dict:
    """Run VLM QA on multi-scene camera collages; does not rewrite assignments."""
    if not config.camera_vlm_collage_qa:
        return {"enabled": False}

    try:
        import cv2
        from src.scene_ocr.config import OpenAIVlmConfig
        from src.scene_ocr.vlm_client import OpenAIVlmClient, parse_vlm_tokens
    except ImportError as exc:
        return {"enabled": True, "error": str(exc), "cameras": []}

    bundle = load_camera_collage_bundle(config.output_dir)
    client = OpenAIVlmClient(OpenAIVlmConfig())
    results: list[dict] = []

    for camera_id in bundle.camera_ids:
        if bundle.scene_count(camera_id) <= 1:
            continue
        collage_path = config.output_dir / "camera_collages" / f"{camera_id.replace('/', '_')}.jpg"
        if not collage_path.is_file():
            continue
        image = cv2.imread(str(collage_path))
        if image is None:
            continue
        tokens = client.read_text([image], prompt=VLM_COLLAGE_PROMPT)
        payload_raw = " ".join(tokens)
        parsed = {}
        try:
            import re

            match = re.search(r"\{.*\}", payload_raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {"raw": payload_raw, "tokens": parse_vlm_tokens(payload_raw)}

        results.append(
            {
                "camera_id": camera_id,
                "scene_count": bundle.scene_count(camera_id),
                "collage_path": str(collage_path),
                "vlm": parsed,
            }
        )

    report = {"enabled": True, "cameras": results}
    out_path = config.artifact("camera_vlm_qa")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
