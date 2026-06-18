"""Resolve frame paths from pipeline CSVs onto the active output directory."""

from __future__ import annotations

from pathlib import Path


def resolve_frame_under_output(path: Path, output_dir: Path) -> Path:
    """Map a CSV frame_path onto output_dir.

    Pipeline artifacts may store host-absolute paths (e.g. from a local run).
    When output_dir is bind-mounted elsewhere (Docker, Cloud Run), remap via
    the ``frames/`` suffix or filename under output_dir.
    """
    output_dir = output_dir.resolve()
    raw = Path(path)

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append((output_dir / raw).resolve())

    if "frames" in raw.parts:
        idx = raw.parts.index("frames")
        candidates.append(output_dir.joinpath(*raw.parts[idx:]).resolve())

    if raw.name:
        candidates.append((output_dir / "frames" / raw.name).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            candidate.relative_to(output_dir)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    for candidate in candidates:
        try:
            candidate.relative_to(output_dir)
            return candidate
        except ValueError:
            continue

    if raw.name:
        return (output_dir / "frames" / raw.name).resolve()
    return (output_dir / raw).resolve()
