# Broadcast Video Pipeline

Turns a sports broadcast `.mp4` into per-camera on-screen text timelines (score bugs, logos, sponsor graphics).

**Start here:** [docs/HANDOFF.md](docs/HANDOFF.md) — complete handoff document covering pipeline stages, design decisions, operations, and debugging.

**Hands-on walkthrough:** [notebooks/broadcast_pipeline_colab_guide.ipynb](notebooks/broadcast_pipeline_colab_guide.ipynb)

**Quick run:**

```bash
pip install -e '.[embedding,ocr]'
python main.py --video data/your_match.mp4 --output-dir data/pipeline
```

Browse results interactively: `python scripts/serve_timeline_viz.py --output-dir data/pipeline`
