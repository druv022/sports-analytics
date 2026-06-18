# Broadcast Video Pipeline

Turns a sports broadcast `.mp4` into per-camera on-screen text timelines (score bugs, logos, sponsor graphics).

**Deep dive:** [docs/HANDOFF.md](docs/HANDOFF.md) — pipeline stages, design decisions, tuning, and debugging.

**Hands-on walkthrough:** [notebooks/broadcast_pipeline_colab_guide.ipynb](notebooks/broadcast_pipeline_colab_guide.ipynb)

Live demo: [https://timeline-viz-815828886655.us-central1.run.app/](https://timeline-viz-815828886655.us-central1.run.app/)

---

## Quick start

### 1. Install

```bash
pip install -e '.[embedding,ocr,appearance,viz]'
python scripts/download_person_seg_model.py   # YOLO11n-seg for appearance stage
```

For GPU (NVIDIA): `pip install -e '.[gpu,appearance,viz]'`

### 2. Run the pipeline

```bash
python main.py --video data/your_match.mp4 --output-dir data/pipeline
```

On CPU-only machines, add `--fast-cameras` for faster iteration (HSV clustering instead of embeddings).

### 3. Review outputs

Primary deliverables in `--output-dir`:

| File | What it is |
|------|------------|
| `aggregated_complete.csv` | Timeline rows with exact text matches |
| `aggregated_partial.csv` | Timeline rows with partial text matches |
| `pipeline_summary.json` | Run stats (scenes, cameras, frame counts) |
| `dropped_text.csv` | OCR tokens that failed association |

Browse interactively:

```bash
python scripts/serve_timeline_viz.py --output-dir data/pipeline
```

Open [http://localhost:8765](http://localhost:8765) — timeline search, frame previews, and camera collages.

---

## Follow-up steps

After a first run, use this checklist to iterate on quality.

### Inspect camera assignments

```bash
python scripts/render_camera_collages.py --output-dir data/pipeline
```

Check `camera_collages/cam_*.jpg`. If cameras look wrong, re-run only the camera stage:

```bash
python main.py --output-dir data/pipeline --from-step cameras --to-step cameras
```

### Curate the text reference catalog

1. Open `data/pipeline/approved_text_reference.csv`.
2. Set `approved` to `true` for strings you want matched; add missing sponsor/bug text.
3. Re-run association and aggregation:

```bash
python main.py --output-dir data/pipeline --from-step associate --resume
```

### Resume or re-run after a crash

```bash
python main.py --output-dir data/pipeline --resume
```

Skip completed stages automatically. To force a specific stage:

```bash
python main.py --output-dir data/pipeline --from-step ocr --to-step ocr
```

Valid step names: `meta`, `extract`, `filter`, `appearance`, `cameras`, `ocr`, `reference`, `enrich`, `associate`, `aggregate`.

### Tune when results look off

| Symptom | Try |
|---------|-----|
| Too many/few scene cuts | Adjust `detector_threshold` in `PipelineConfig` (default 27) |
| Too many camera IDs | Raise `camera_merge_similarity_threshold` or check collages |
| High `dropped_text` rate | Edit reference CSV; consider `--enable-vlm` for hard OCR |
| OCR too slow | Reduce `--ocr-samples-per-sec` or install GPU extras |

See [docs/HANDOFF.md §8](docs/HANDOFF.md) for the full tuning guide.

### Deploy the timeline viewer (optional)

A Docker image serves the viz UI with pipeline data baked in:

```bash
docker build --platform linux/amd64 \
  -f docker/viz/Dockerfile \
  --build-arg PIPELINE_SRC=data/pipeline \
  -t timeline-viz .
```

See `docker/viz/Dockerfile` and [docs/HANDOFF.md §7](docs/HANDOFF.md) for Cloud Run deployment details.

---

## Useful scripts

| Script | Purpose |
|--------|---------|
| `scripts/serve_timeline_viz.py` | Interactive timeline + camera collage UI |
| `scripts/render_camera_collages.py` | Export per-camera thumbnail grids |
| `scripts/rerun_cameras.py` | Re-cluster cameras with before/after metrics |
| `scripts/run_ocr.py` | Test OCR on a single image |
| `scripts/benchmark_ocr.py` | Measure OCR throughput |
