from __future__ import annotations


def parse_frame_ranges(value: str) -> list[int]:
    """Parse a semicolon-separated frame range string into sorted frame numbers.

    Examples:
        "10-12;20" -> [10, 11, 12, 20]
        "5" -> [5]
    """
    if not value or not str(value).strip():
        return []

    frames: list[int] = []
    for part in str(value).split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                start, end = end, start
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(part))

    return sorted(set(frames))
