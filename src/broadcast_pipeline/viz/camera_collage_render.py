from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from broadcast_pipeline.viz.camera_collage import (
    CameraCollageBundle,
    SceneCollageEntry,
    SceneFrameSlot,
    SlotName,
    load_camera_collage_bundle,
)

LayoutName = Literal["timeline", "grid"]


@dataclass(frozen=True)
class CollageRenderConfig:
    thumb_width: int = 320
    thumb_height: int = 180
    padding: int = 8
    label_height: int = 22
    header_height: int = 36
    background: str = "#1a1a1a"
    slots: tuple[SlotName, ...] = ("mid",)
    layout: LayoutName = "timeline"
    grid_columns: int = 4


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        font_path = Path(path)
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _resolve_frame_path(path: Path, output_dir: Path) -> Path:
    if path.is_file():
        return path.resolve()
    candidate = (output_dir / path).resolve()
    if candidate.is_file():
        return candidate
    return path


def _load_thumbnail(path: Path, output_dir: Path, size: tuple[int, int]) -> Image.Image:
    resolved = _resolve_frame_path(path, output_dir)
    if not resolved.is_file():
        placeholder = Image.new("RGB", size, color="#333333")
        draw = ImageDraw.Draw(placeholder)
        draw.text((8, 8), "missing", fill="#aaaaaa", font=_load_font(12))
        return placeholder

    with Image.open(resolved) as img:
        rgb = img.convert("RGB")
    rgb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color="#222222")
    offset = ((size[0] - rgb.width) // 2, (size[1] - rgb.height) // 2)
    canvas.paste(rgb, offset)
    return canvas


def _scene_label(entry: SceneCollageEntry) -> str:
    if entry.start_sec is not None and entry.end_sec is not None:
        return f"scene {entry.scene_id}  {entry.start_sec:.1f}s–{entry.end_sec:.1f}s"
    return f"scene {entry.scene_id}"


def _pick_slots(entry: SceneCollageEntry, slots: tuple[SlotName, ...]) -> list[SceneFrameSlot]:
    by_name = {slot.slot: slot for slot in entry.frames}
    return [by_name[name] for name in slots if name in by_name]


def _timeline_canvas_size(
    entries: list[SceneCollageEntry],
    config: CollageRenderConfig,
) -> tuple[int, int]:
    thumb_size = (config.thumb_width, config.thumb_height)
    n_slots = max(len(config.slots), 1)
    row_width = config.padding + n_slots * (thumb_size[0] + config.padding)
    row_height = config.label_height + thumb_size[1] + config.padding
    width = row_width + config.padding
    height = config.header_height + len(entries) * row_height + config.padding
    return width, height


def _grid_canvas_size(n_entries: int, config: CollageRenderConfig) -> tuple[int, int]:
    cols = max(1, min(config.grid_columns, n_entries or 1))
    rows = (max(n_entries, 1) + cols - 1) // cols
    cell_w = config.thumb_width + config.padding
    cell_h = config.thumb_height + config.label_height + config.padding
    width = config.padding + cols * cell_w + config.padding
    height = config.header_height + rows * cell_h + config.padding
    return width, height


def render_camera_collage_image(
    bundle: CameraCollageBundle,
    camera_id: str,
    config: CollageRenderConfig | None = None,
) -> Image.Image:
    """Render one verification collage for a single camera."""
    config = config or CollageRenderConfig()
    entries = bundle.scenes_for_camera(camera_id)
    thumb_size = (config.thumb_width, config.thumb_height)
    font = _load_font(13)
    header_font = _load_font(16)

    if config.layout == "grid":
        width, height = _grid_canvas_size(len(entries), config)
    else:
        width, height = _timeline_canvas_size(entries, config)

    canvas = Image.new("RGB", (width, height), color=config.background)
    draw = ImageDraw.Draw(canvas)
    title = f"{camera_id}  ({len(entries)} scene{'s' if len(entries) != 1 else ''})"
    draw.text((config.padding, 8), title, fill="#f2f2f2", font=header_font)

    if not entries:
        draw.text(
            (config.padding, config.header_height),
            "No scenes assigned",
            fill="#bbbbbb",
            font=font,
        )
        return canvas

    y = config.header_height
    if config.layout == "grid":
        cols = max(1, min(config.grid_columns, len(entries)))
        cell_w = config.thumb_width + config.padding
        cell_h = config.thumb_height + config.label_height + config.padding
        for idx, entry in enumerate(entries):
            col = idx % cols
            row = idx // cols
            x = config.padding + col * cell_w
            cell_y = y + row * cell_h
            slot = _pick_slots(entry, config.slots)
            frame = slot[0] if slot else None
            if frame is None:
                continue
            thumb = _load_thumbnail(Path(frame.frame_path), bundle.output_dir, thumb_size)
            canvas.paste(thumb, (x, cell_y))
            draw.text(
                (x, cell_y + thumb_size[1] + 2),
                _scene_label(entry),
                fill="#dddddd",
                font=font,
            )
        return canvas

    row_height = thumb_size[1] + config.label_height + config.padding
    for entry in entries:
        draw.text((config.padding, y + 2), _scene_label(entry), fill="#dddddd", font=font)
        x = config.padding
        frame_y = y + config.label_height
        for slot in _pick_slots(entry, config.slots):
            thumb = _load_thumbnail(Path(slot.frame_path), bundle.output_dir, thumb_size)
            canvas.paste(thumb, (x, frame_y))
            draw.text(
                (x + 4, frame_y + thumb_size[1] - 16),
                slot.slot,
                fill="#ffffff",
                font=_load_font(11),
            )
            x += thumb_size[0] + config.padding
        y += row_height

    return canvas


def render_camera_collages(
    output_dir: Path,
    dest_dir: Path | None = None,
    config: CollageRenderConfig | None = None,
) -> list[Path]:
    """Write one collage JPEG per camera under dest_dir (default: output_dir/camera_collages)."""
    config = config or CollageRenderConfig()
    bundle = load_camera_collage_bundle(output_dir)
    out_dir = dest_dir or (Path(output_dir) / "camera_collages")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for camera_id in bundle.camera_ids:
        image = render_camera_collage_image(bundle, camera_id, config=config)
        safe_name = camera_id.replace("/", "_")
        path = out_dir / f"{safe_name}.jpg"
        image.save(path, format="JPEG", quality=90, optimize=True)
        written.append(path)
    return written
