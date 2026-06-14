"""Vendored TennisCourtDetector inference code (yastrebksv/TennisCourtDetector)."""

__all__ = ["BallTrackerNet", "CourtReference"]


def __getattr__(name: str):
    if name == "BallTrackerNet":
        from .tracknet import BallTrackerNet
        return BallTrackerNet
    if name == "CourtReference":
        from .court_reference import CourtReference
        return CourtReference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
