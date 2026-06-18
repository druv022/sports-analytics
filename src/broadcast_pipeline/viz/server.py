from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from broadcast_pipeline.camera_debug import load_camera_clustering_debug
from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.viz.camera_collage import (
    CameraCollageBundle,
    CameraCollageLoadError,
    load_camera_collage_bundle,
    scene_entry_to_dict,
)
from broadcast_pipeline.viz.frame_paths import resolve_frame_under_output
from broadcast_pipeline.viz.camera_compare import (
    SceneSelection,
    _scene_ids_for_cameras,
    build_global_projection,
    compare_scenes,
)
from broadcast_pipeline.viz.appearance_api import run_appearance_on_bytes
from broadcast_pipeline.viz.appearance_loader import (
    AppearanceBundle,
    load_appearance_bundle,
    scene_row_to_dict,
)
from broadcast_pipeline.viz.data_loader import TimelineBundle, TimelineLoadError, load_timeline_bundle
from broadcast_pipeline.viz.pipeline_ocr import (
    PipelineOcrIndex,
    PipelineOcrLoadError,
    load_pipeline_ocr_index,
)


class CompareSelection(BaseModel):
    camera_id: str
    scene_id: int


class CompareRequest(BaseModel):
    selections: list[CompareSelection] = Field(min_length=2)
    include_global: bool = False


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith((".js", ".css", ".html", ".map")):
            for key, value in _NO_CACHE_HEADERS.items():
                response.headers[key] = value
        return response


def _asset_version(static_dir: Path) -> str:
    mtimes = [
        path.stat().st_mtime
        for path in static_dir.iterdir()
        if path.is_file() and path.suffix in {".js", ".css", ".html"}
    ]
    if not mtimes:
        return "0"
    return str(int(max(mtimes)))


def _artifact_fingerprint(output_dir: Path, names: tuple[str, ...]) -> str:
    mtimes: list[float] = []
    for name in names:
        path = output_dir / name
        mtimes.append(path.stat().st_mtime if path.is_file() else 0.0)
    return str(int(max(mtimes)))


_CAMERA_BUNDLE_ARTIFACTS = (
    "scene_assignments.csv",
    "frame_index.csv",
    "scenes.csv",
    "camera_assignment_analysis.json",
)
_TIMELINE_BUNDLE_ARTIFACTS = (
    "aggregated_complete.csv",
    "aggregated_partial.csv",
    "frame_index.csv",
    "frame_assignments.csv",
    "frame_text_associated.csv",
)
_PIPELINE_OCR_ARTIFACTS = (
    "frame_ocr.csv",
    "frame_text_associated.csv",
)


def create_app(output_dir: Path, static_dir: Path) -> FastAPI:
    app = FastAPI(title="Timeline Visualization", version="0.1.0")
    bundle: TimelineBundle | None = None
    bundle_fingerprint: str | None = None
    camera_bundle: CameraCollageBundle | None = None
    camera_bundle_fingerprint: str | None = None
    appearance_bundle: AppearanceBundle | None = None
    pipeline_ocr: PipelineOcrIndex | None = None
    pipeline_ocr_fingerprint: str | None = None

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"https?://("
            r"127\.0\.0\.1|localhost"
            r"|[^/]+\.colab\.(?:research\.google\.com|google\.com)"
            r"|colab\.(?:research\.google\.com|google\.com)"
            r")(:\d+)?"
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def disable_client_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/assets/"):
            for key, value in _NO_CACHE_HEADERS.items():
                response.headers[key] = value
        return response

    def get_bundle() -> TimelineBundle:
        nonlocal bundle, bundle_fingerprint
        fingerprint = _artifact_fingerprint(output_dir, _TIMELINE_BUNDLE_ARTIFACTS)
        if bundle is None or bundle_fingerprint != fingerprint:
            try:
                bundle = load_timeline_bundle(output_dir)
            except TimelineLoadError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            bundle_fingerprint = fingerprint
        return bundle

    def get_camera_bundle() -> CameraCollageBundle:
        nonlocal camera_bundle, camera_bundle_fingerprint
        fingerprint = _artifact_fingerprint(output_dir, _CAMERA_BUNDLE_ARTIFACTS)
        if camera_bundle is None or camera_bundle_fingerprint != fingerprint:
            try:
                camera_bundle = load_camera_collage_bundle(output_dir)
            except CameraCollageLoadError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            camera_bundle_fingerprint = fingerprint
        return camera_bundle

    def get_appearance_bundle() -> AppearanceBundle:
        nonlocal appearance_bundle
        if appearance_bundle is None:
            appearance_bundle = load_appearance_bundle(output_dir)
        return appearance_bundle

    def get_pipeline_ocr() -> PipelineOcrIndex:
        nonlocal pipeline_ocr, pipeline_ocr_fingerprint
        fingerprint = _artifact_fingerprint(output_dir, _PIPELINE_OCR_ARTIFACTS)
        if pipeline_ocr is None or pipeline_ocr_fingerprint != fingerprint:
            try:
                pipeline_ocr = load_pipeline_ocr_index(output_dir)
            except PipelineOcrLoadError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            pipeline_ocr_fingerprint = fingerprint
        return pipeline_ocr

    def _resolve_camera_frame_path(frame_number: int) -> Path:
        appearance = get_appearance_bundle()
        frame_row = appearance.frames_by_number.get(frame_number)
        if frame_row is not None and frame_row.frame_path:
            path = Path(frame_row.frame_path)
            if not path.is_absolute():
                path = (appearance.output_dir / path).resolve()
            else:
                path = path.resolve()
            return _guard_output_path(path, appearance.output_dir.resolve())

        try:
            camera_data = get_camera_bundle()
        except HTTPException:
            camera_data = None

        if camera_data is not None:
            import pandas as pd

            frame_index_path = camera_data.output_dir / "frame_index.csv"
            if frame_index_path.is_file():
                frame_index = pd.read_csv(frame_index_path)
                camera_rows = frame_index[
                    (frame_index["sample_role"] == "camera")
                    & (frame_index["frame_number"] == frame_number)
                ]
                if not camera_rows.empty:
                    row = camera_rows.iloc[0]
                    path = Path(str(row.frame_path))
                    if not path.is_absolute():
                        path = (camera_data.output_dir / path).resolve()
                    else:
                        path = path.resolve()
                    return _guard_output_path(path, camera_data.output_dir.resolve())

        return _resolve_frame_path(frame_number)

    def _run_segmentation_on_path(path: Path, *, scene_id: int, frame_number: int) -> dict:
        try:
            return run_appearance_on_bytes(
                path.read_bytes(),
                output_dir=output_dir,
                scene_id=scene_id,
                frame_number=frame_number,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc

    def _guard_output_path(path: Path, output_root: Path) -> Path:
        resolved = resolve_frame_under_output(path, output_root)
        try:
            resolved.relative_to(output_root.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Invalid frame path") from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="Frame image missing on disk")
        return resolved

    @app.get("/api/timeline")
    def api_timeline() -> dict:
        data = get_bundle()
        return {
            "output_dir": str(data.output_dir),
            "summary": data.summary,
            "suggestions": data.suggestions,
            "rows": data.rows,
        }

    @app.get("/api/search")
    def api_search(q: str = Query(default="")) -> dict:
        data = get_bundle()
        return {
            "query": q,
            "suggestions": data.search_suggestions(q),
            "rows": data.search_rows(q),
        }

    @app.get("/api/row")
    def api_row(
        camera_id: str = Query(...),
        mapped: str = Query(...),
        text: str | None = Query(default=None),
    ) -> dict:
        data = get_bundle()
        row = data.find_row(camera_id, mapped, text)
        if row is None:
            raise HTTPException(status_code=404, detail="Row not found")
        return data.row_detail(row)

    def _resolve_frame_path(frame_number: int) -> Path:
        data = get_bundle()
        info = data.frame_lookup.get(frame_number)
        if info is None or not info.frame_path:
            raise HTTPException(status_code=404, detail="Frame not found")

        path = Path(info.frame_path).resolve()
        output_root = data.output_dir.resolve()
        return _guard_output_path(path, output_root)

    @app.get("/api/cameras")
    def api_cameras() -> dict:
        data = get_camera_bundle()
        counts = {camera_id: data.scene_count(camera_id) for camera_id in data.camera_ids}
        return {
            "camera_ids": data.camera_ids,
            "counts": counts,
            "has_debug_artifact": data.has_debug_artifact,
            "camera_cluster_ids": data.camera_cluster_ids or {},
        }

    @app.get("/api/cameras/{camera_id}/scenes")
    def api_camera_scenes(camera_id: str) -> dict:
        data = get_camera_bundle()
        if camera_id not in data.scenes_by_camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        scenes = [scene_entry_to_dict(entry) for entry in data.scenes_for_camera(camera_id)]
        return {"camera_id": camera_id, "scenes": scenes}

    @app.get("/api/scene-images/{scene_id}/{slot}")
    def api_scene_image(scene_id: int, slot: str) -> FileResponse:
        if slot not in {"begin", "mid", "end"}:
            raise HTTPException(status_code=400, detail="Invalid slot")
        data = get_camera_bundle()
        path = data.resolve_slot_path(scene_id, slot)  # type: ignore[arg-type]
        if path is None:
            raise HTTPException(status_code=404, detail="Scene slot not found")
        return FileResponse(_guard_output_path(path, data.output_dir.resolve()))

    @app.post("/api/cameras/compare")
    def api_cameras_compare(body: CompareRequest) -> dict:
        try:
            selections = [
                SceneSelection(camera_id=item.camera_id, scene_id=item.scene_id)
                for item in body.selections
            ]
            return compare_scenes(output_dir, selections, include_global=body.include_global)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/camera-debug/global")
    def api_camera_debug_global(
        scene_id: list[int] = Query(default=[]),
        camera_id: list[str] = Query(default=[]),
    ) -> dict:
        config = PipelineConfig(output_dir=output_dir)
        debug = load_camera_clustering_debug(config.artifact("camera_clustering_debug"))
        if debug is None:
            raise HTTPException(
                status_code=404,
                detail="camera_clustering_debug.npz not found. Re-run the cameras stage.",
            )
        highlight_scene_ids: list[int] | None = scene_id or None
        if camera_id:
            scene_assignments_path = config.artifact("scene_assignments")
            if scene_assignments_path.is_file():
                import pandas as pd

                scene_assignments = pd.read_csv(scene_assignments_path)
                highlight_scene_ids = _scene_ids_for_cameras(scene_assignments, set(camera_id))
        return build_global_projection(
            debug,
            highlight_scene_ids=highlight_scene_ids,
            output_dir=output_dir,
        )

    @app.get("/api/camera-debug/analysis")
    def api_camera_debug_analysis() -> dict:
        path = output_dir / "camera_assignment_analysis.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="camera_assignment_analysis.json not found")
        import json

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="Invalid analysis JSON") from exc

    @app.get("/api/frames/{frame_number}")
    def api_frame(frame_number: int) -> FileResponse:
        return FileResponse(_resolve_frame_path(frame_number))

    @app.get("/api/pipeline/ocr/{frame_number}")
    def api_pipeline_ocr_frame(frame_number: int) -> dict:
        data = get_pipeline_ocr()
        try:
            return data.frame_payload(frame_number)
        except PipelineOcrLoadError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/appearance")
    def api_appearance() -> dict:
        data = get_appearance_bundle()
        groups = data.compatibility_groups()
        grouped: dict[int, list[int]] = {}
        for scene_id, component_id in groups.items():
            grouped.setdefault(component_id, []).append(scene_id)
        return {
            "output_dir": str(data.output_dir),
            "has_artifacts": data.has_appearance_artifacts,
            "summary": data.build_summary() if data.has_appearance_artifacts else {},
            "compatibility_groups": {
                str(component_id): sorted(scene_ids)
                for component_id, scene_ids in sorted(grouped.items())
            },
            "scenes_with_issues": data.scenes_with_issues() if data.has_appearance_artifacts else [],
        }

    @app.get("/api/appearance/scenes")
    def api_appearance_scenes() -> dict:
        data = get_appearance_bundle()
        if not data.has_appearance_artifacts:
            return {"scenes": [], "has_artifacts": False}
        return {
            "has_artifacts": True,
            "scenes": [scene_row_to_dict(scene) for scene in data.scene_list()],
        }

    @app.get("/api/appearance/scene/{scene_id}")
    def api_appearance_scene(scene_id: int) -> dict:
        data = get_appearance_bundle()
        if not data.has_appearance_artifacts:
            raise HTTPException(
                status_code=404,
                detail="Appearance artifacts not found. Run the pipeline through the appearance stage.",
            )
        detail = data.scene_detail(scene_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Scene not found")
        return detail

    @app.get("/api/appearance/segment/frame/{frame_number}")
    def api_appearance_segment_frame(frame_number: int) -> dict:
        path = _resolve_camera_frame_path(frame_number)
        appearance = get_appearance_bundle()
        frame_row = appearance.frames_by_number.get(frame_number)
        scene_id = frame_row.scene_id if frame_row is not None else -1
        return _run_segmentation_on_path(path, scene_id=scene_id, frame_number=frame_number)

    @app.get("/api/appearance/segment/scene/{scene_id}/{slot}")
    def api_appearance_segment_scene(scene_id: int, slot: str) -> dict:
        if slot not in {"begin", "mid", "end"}:
            raise HTTPException(status_code=400, detail="Invalid slot")
        try:
            camera_data = get_camera_bundle()
        except HTTPException as exc:
            raise HTTPException(
                status_code=503,
                detail="Camera collage artifacts required for scene slot segmentation.",
            ) from exc
        path = camera_data.resolve_slot_path(scene_id, slot)  # type: ignore[arg-type]
        if path is None:
            raise HTTPException(status_code=404, detail="Scene slot not found")
        guarded = _guard_output_path(path, camera_data.output_dir.resolve())
        slot_info = camera_data.slot_lookup.get((scene_id, slot))  # type: ignore[arg-type]
        frame_number = slot_info.frame_number if slot_info is not None else -1
        return _run_segmentation_on_path(guarded, scene_id=scene_id, frame_number=frame_number)

    index_path = static_dir / "index.html"
    if index_path.is_file():
        asset_version = _asset_version(static_dir)
        app.mount("/assets", NoCacheStaticFiles(directory=static_dir), name="assets")

        @app.get("/")
        def index() -> HTMLResponse:
            content = index_path.read_text(encoding="utf-8")
            content = content.replace("__ASSET_VERSION__", asset_version)
            return HTMLResponse(content, headers=_NO_CACHE_HEADERS)

    return app
