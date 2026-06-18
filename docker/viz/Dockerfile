# Timeline viz — standalone container for Google Cloud Run (or any Docker host).
#
# Build locally for Cloud Run (must be linux/amd64 — add --platform on Apple Silicon):
#   docker build --platform linux/amd64 -f docker/viz/Dockerfile \
#     --build-arg PIPELINE_SRC=data/pipeline \
#     -t us-central1-docker.pkg.dev/porfolio-demos/timeline-viz/timeline-viz:latest .
#
# Or build on GCP (recommended on Mac — always amd64):
#   gcloud builds submit --project porfolio-demos \
#     --config docker/viz/cloudbuild.yaml \
#     --substitutions=_PIPELINE_SRC=data/pipeline
#
# Run locally (native arch; mount pipeline output):
#   docker run --rm -p 8080:8080 \
#     -v "$(pwd)/data/pipeline:/app/data/pipeline:ro" \
#     timeline-viz
#
# Cloud Run deploy (after pushing to Artifact Registry):
#   gcloud run deploy timeline-viz \
#     --project porfolio-demos \
#     --image us-central1-docker.pkg.dev/porfolio-demos/timeline-viz/timeline-viz:latest \
#     --region us-central1 \
#     --port 8080 \
#     --memory 2Gi \
#     --cpu 2 \
#     --timeout 300 \
#     --allow-unauthenticated

# -----------------------------------------------------------------------------
# Stage 1: export YOLO person-segmentation ONNX (build-only; not in final image)
# -----------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS model-builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY docker/viz/export_person_seg.py docker/viz/export_person_seg.py

RUN pip install ultralytics onnx \
    && python docker/viz/export_person_seg.py --output /models/person_seg/yolo11n-seg.onnx

# -----------------------------------------------------------------------------
# Stage 2: runtime image
# -----------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    OUTPUT_DIR=/app/data/pipeline \
    STATIC_DIR=/app/static/timeline_viz \
    PORT=8080 \
    HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY static/timeline_viz/ static/timeline_viz/
COPY docker/viz/requirements.txt docker/viz/requirements.txt
COPY docker/viz/prewarm_models.py docker/viz/prewarm_models.py
COPY docker/viz/entrypoint.sh /entrypoint.sh

RUN pip install --no-cache-dir -r docker/viz/requirements.txt \
    && pip install --no-cache-dir --no-deps -e . \
    && sed -i 's/\r$//' /entrypoint.sh \
    && chmod +x /entrypoint.sh \
    && mkdir -p /app/data/pipeline

COPY --from=model-builder /models/person_seg/yolo11n-seg.onnx models/person_seg/yolo11n-seg.onnx

RUN python docker/viz/prewarm_models.py

# Optional pipeline bundle (CSVs + frames). Default is an empty placeholder dir.
ARG PIPELINE_SRC=docker/viz/empty-pipeline
COPY ${PIPELINE_SRC}/ /app/data/pipeline/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=3)"

ENTRYPOINT ["/entrypoint.sh"]
