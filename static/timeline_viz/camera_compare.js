function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const GLOBAL_PAD = 36;
const GLOBAL_MIN_ZOOM = 0.4;
const GLOBAL_MAX_ZOOM = 12;

const GLOBAL_CANVAS_ASPECT = 420 / 900;

const globalProjectionState = {
  projection: null,
  bounds: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStartX: 0,
  dragStartY: 0,
  panStartX: 0,
  panStartY: 0,
  hoveredPoint: null,
  lastPointer: null,
  controlsBound: false,
  resizeObserver: null,
  resizeListener: null,
};

function pointKey(point) {
  if (!point) return "";
  return `${point.scene_id}:${point.frame_number}`;
}

function findPointAt(canvasX, canvasY) {
  const canvas = document.getElementById("camera-global-canvas");
  const projection = globalProjectionState.projection;
  if (!canvas || !projection?.points?.length || !globalProjectionState.bounds) return null;

  const width = canvas.width;
  const height = canvas.height;
  let best = null;
  let bestDist = Infinity;

  projection.points.forEach((point) => {
    const { x, y } = dataToScreen(point.x, point.y, width, height);
    const radius = point.highlighted ? 6 : 3.5;
    const hitRadius = Math.max(10, radius + 5);
    const dist = Math.hypot(canvasX - x, canvasY - y);
    if (dist <= hitRadius && dist < bestDist) {
      bestDist = dist;
      best = point;
    }
  });

  return best;
}

function formatPointTooltipHtml(point) {
  const clusterLabel = point.cluster_id < 0 ? "noise (-1)" : String(point.cluster_id);
  const preMergeCluster =
    point.pre_merge_cluster_id < 0 ? "noise (-1)" : String(point.pre_merge_cluster_id);
  const selected = point.highlighted ? " · on selected camera" : "";
  return `
    <strong>Scene ${escapeHtml(point.scene_id)} · frame ${escapeHtml(point.frame_number)}${escapeHtml(selected)}</strong>
    <dl>
      <dt>Frame camera</dt><dd>${escapeHtml(point.camera_id)}</dd>
      <dt>Frame cluster</dt><dd>${escapeHtml(clusterLabel)}</dd>
      <dt>Scene camera</dt><dd>${escapeHtml(point.scene_camera_id)}</dd>
      <dt>Scene cluster</dt><dd>${escapeHtml(String(point.scene_cluster_id))}</dd>
      <dt>Pre-merge cluster</dt><dd>${escapeHtml(preMergeCluster)}</dd>
      <dt>Pre-merge camera</dt><dd>${escapeHtml(point.pre_merge_camera_id)}</dd>
    </dl>`;
}

function hideGlobalTooltip() {
  const tooltip = document.getElementById("camera-global-tooltip");
  if (!tooltip) return;
  tooltip.hidden = true;
  tooltip.classList.add("hidden");
  tooltip.innerHTML = "";
}

function showGlobalTooltip(point, clientX, clientY, wrap) {
  const tooltip = document.getElementById("camera-global-tooltip");
  if (!tooltip) return;

  tooltip.innerHTML = formatPointTooltipHtml(point);
  tooltip.hidden = false;
  tooltip.classList.remove("hidden");

  const wrapRect = wrap.getBoundingClientRect();
  let left = clientX - wrapRect.left + 12;
  let top = clientY - wrapRect.top + 12;
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;

  const tipRect = tooltip.getBoundingClientRect();
  if (left + tipRect.width > wrapRect.width - 8) {
    left = clientX - wrapRect.left - tipRect.width - 12;
  }
  if (top + tipRect.height > wrapRect.height - 8) {
    top = clientY - wrapRect.top - tipRect.height - 12;
  }
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function setHoveredPoint(point) {
  const nextKey = pointKey(point);
  if (pointKey(globalProjectionState.hoveredPoint) === nextKey) return;
  globalProjectionState.hoveredPoint = point;
  renderGlobalProjectionCanvas();
}

function clampGlobalZoom(value) {
  return Math.min(GLOBAL_MAX_ZOOM, Math.max(GLOBAL_MIN_ZOOM, value));
}

function updateGlobalZoomLabel() {
  const label = document.getElementById("camera-global-zoom-label");
  if (label) {
    label.textContent = `${Math.round(globalProjectionState.zoom * 100)}%`;
  }
}

function resetGlobalView() {
  globalProjectionState.zoom = 1;
  globalProjectionState.panX = 0;
  globalProjectionState.panY = 0;
  updateGlobalZoomLabel();
}

function canvasPointerCoords(canvas, clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function zoomGlobalAt(factor, mx, my) {
  const canvas = document.getElementById("camera-global-canvas");
  if (!canvas) return;

  const oldZoom = globalProjectionState.zoom;
  const newZoom = clampGlobalZoom(oldZoom * factor);
  if (newZoom === oldZoom) return;

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  globalProjectionState.panX = mx - cx - (mx - cx - globalProjectionState.panX) * (newZoom / oldZoom);
  globalProjectionState.panY = my - cy - (my - cy - globalProjectionState.panY) * (newZoom / oldZoom);
  globalProjectionState.zoom = newZoom;
  updateGlobalZoomLabel();
}

function computeProjectionBounds(points) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function projectionBaseScale(bounds, width, height) {
  const dataW = bounds.maxX - bounds.minX || 1;
  const dataH = bounds.maxY - bounds.minY || 1;
  const plotW = width - GLOBAL_PAD * 2;
  const plotH = height - GLOBAL_PAD * 2;
  return Math.min(plotW / dataW, plotH / dataH);
}

function dataToScreen(x, y, width, height) {
  const { bounds, zoom, panX, panY } = globalProjectionState;
  const scale = projectionBaseScale(bounds, width, height);
  const dataCx = (bounds.minX + bounds.maxX) / 2;
  const dataCy = (bounds.minY + bounds.maxY) / 2;
  const cx = width / 2;
  const cy = height / 2;
  return {
    x: cx + (x - dataCx) * scale * zoom + panX,
    y: cy + (y - dataCy) * scale * zoom + panY,
  };
}

function syncGlobalCanvasSize() {
  const wrap = document.getElementById("camera-global-canvas-wrap");
  const canvas = document.getElementById("camera-global-canvas");
  if (!wrap || !canvas) return false;

  const width = Math.max(320, Math.floor(wrap.clientWidth));
  const height = Math.max(280, Math.round(width * GLOBAL_CANVAS_ASPECT));
  if (canvas.width === width && canvas.height === height) return false;

  canvas.width = width;
  canvas.height = height;
  canvas.style.height = `${height}px`;
  return true;
}

function setupGlobalCanvasResize() {
  const wrap = document.getElementById("camera-global-canvas-wrap");
  if (!wrap || globalProjectionState.resizeObserver) return;

  const onResize = () => {
    if (syncGlobalCanvasSize() && globalProjectionState.projection) {
      renderGlobalProjectionCanvas();
    }
  };

  if (typeof ResizeObserver !== "undefined") {
    globalProjectionState.resizeObserver = new ResizeObserver(onResize);
    globalProjectionState.resizeObserver.observe(wrap);
  } else {
    window.addEventListener("resize", onResize);
    globalProjectionState.resizeListener = onResize;
  }
}

function setupGlobalProjectionControls() {
  if (globalProjectionState.controlsBound) return;
  globalProjectionState.controlsBound = true;

  const wrap = document.getElementById("camera-global-canvas-wrap");
  const canvas = document.getElementById("camera-global-canvas");
  if (!wrap || !canvas) return;

  document.getElementById("camera-global-zoom-in")?.addEventListener("click", () => {
    zoomGlobalAt(1.2, canvas.width / 2, canvas.height / 2);
    renderGlobalProjectionCanvas();
  });
  document.getElementById("camera-global-zoom-out")?.addEventListener("click", () => {
    zoomGlobalAt(1 / 1.2, canvas.width / 2, canvas.height / 2);
    renderGlobalProjectionCanvas();
  });
  document.getElementById("camera-global-zoom-reset")?.addEventListener("click", () => {
    resetGlobalView();
    renderGlobalProjectionCanvas();
  });

  wrap.addEventListener(
    "wheel",
    (event) => {
      if (!globalProjectionState.projection?.points?.length) return;
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      const anchor = canvasPointerCoords(canvas, event.clientX, event.clientY);
      zoomGlobalAt(factor, anchor.x, anchor.y);
      renderGlobalProjectionCanvas();
    },
    { passive: false },
  );

  wrap.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || !globalProjectionState.projection?.points?.length) return;
    hideGlobalTooltip();
    globalProjectionState.dragging = true;
    globalProjectionState.dragStartX = event.clientX;
    globalProjectionState.dragStartY = event.clientY;
    globalProjectionState.panStartX = globalProjectionState.panX;
    globalProjectionState.panStartY = globalProjectionState.panY;
    wrap.classList.add("dragging");
    wrap.classList.remove("hovering-point");
  });

  wrap.addEventListener("mousemove", (event) => {
    globalProjectionState.lastPointer = {
      clientX: event.clientX,
      clientY: event.clientY,
    };
    if (globalProjectionState.dragging) return;
    if (!globalProjectionState.projection?.points?.length) {
      hideGlobalTooltip();
      return;
    }
    const anchor = canvasPointerCoords(canvas, event.clientX, event.clientY);
    const point = findPointAt(anchor.x, anchor.y);
    setHoveredPoint(point);
    if (point) {
      showGlobalTooltip(point, event.clientX, event.clientY, wrap);
      wrap.classList.add("hovering-point");
    } else {
      hideGlobalTooltip();
      wrap.classList.remove("hovering-point");
    }
  });

  wrap.addEventListener("mouseleave", () => {
    globalProjectionState.hoveredPoint = null;
    hideGlobalTooltip();
    wrap.classList.remove("hovering-point");
    renderGlobalProjectionCanvas();
  });

  setupGlobalCanvasResize();

  window.addEventListener("mousemove", (event) => {
    if (!globalProjectionState.dragging) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    globalProjectionState.panX =
      globalProjectionState.panStartX + (event.clientX - globalProjectionState.dragStartX) * scaleX;
    globalProjectionState.panY =
      globalProjectionState.panStartY + (event.clientY - globalProjectionState.dragStartY) * scaleY;
    renderGlobalProjectionCanvas();
  });

  window.addEventListener("mouseup", () => {
    if (!globalProjectionState.dragging) return;
    globalProjectionState.dragging = false;
    wrap.classList.remove("dragging");
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setupGlobalProjectionControls);
} else {
  setupGlobalProjectionControls();
}

function formatPct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

function formatFloat(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function sceneLabel(scene) {
  const parts = [`Scene ${scene.scene_id}`];
  if (scene.start_sec != null && scene.end_sec != null) {
    parts.push(`${Number(scene.start_sec).toFixed(1)}s–${Number(scene.end_sec).toFixed(1)}s`);
  }
  parts.push(`cluster ${scene.cluster_id}`);
  parts.push(`${formatPct(scene.winner_share)} votes`);
  return parts.join(" · ");
}

function voteBarHtml(votes, winner) {
  const total = Object.values(votes || {}).reduce((sum, n) => sum + Number(n), 0);
  if (!total) return "";
  const colors = ["#5b9dff", "#3ecf8e", "#f0b429", "#ff6b6b", "#b794f4", "#63b3ed"];
  let idx = 0;
  const segments = Object.entries(votes)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([cameraId, count]) => {
      const width = (Number(count) / total) * 100;
      const color = cameraId === winner ? colors[0] : colors[(idx++ % (colors.length - 1)) + 1];
      return `<span title="${escapeHtml(cameraId)}: ${count}" style="width:${width}%;background:${color}"></span>`;
    })
    .join("");
  return `<div class="vote-bar">${segments}</div>`;
}

function renderPairwiseTable(payload) {
  const container = document.getElementById("camera-compare-pairwise");
  if (!container) return;
  const scenes = payload.per_scene || [];
  const rows = payload.pairwise || [];
  if (!rows.length) {
    container.innerHTML = "<p class='muted'>No pairwise metrics available.</p>";
    return;
  }

  const header = ["Pair", "Mean cosine", "Mean co-assoc", "Same cluster", "Same camera", "Verdict"];
  const body = rows.map((row) => {
    const a = scenes[row.a];
    const b = scenes[row.b];
    return `
      <tr>
        <td>${escapeHtml(`${a.camera_id} s${a.scene_id}`)} vs ${escapeHtml(`${b.camera_id} s${b.scene_id}`)}</td>
        <td>${formatFloat(row.mean_cosine)}</td>
        <td>${formatFloat(row.mean_coassoc)}</td>
        <td>${row.same_cluster_id ? "yes" : "no"}</td>
        <td>${row.same_camera_id ? "yes" : "no"}</td>
        <td class="verdict-${escapeHtml(row.verdict)}">${escapeHtml(row.verdict)}</td>
      </tr>`;
  });

  container.innerHTML = `
    <table>
      <thead><tr>${header.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${body.join("")}</tbody>
    </table>`;
}

function renderExplanations(payload) {
  const container = document.getElementById("camera-compare-explanations");
  if (!container) return;
  const lines = payload.explanations || [];
  if (!lines.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = lines
    .map((line) => `<div class="compare-explanation">${escapeHtml(line)}</div>`)
    .join("");
}

function renderSceneCards(payload) {
  const container = document.getElementById("camera-compare-scenes");
  if (!container) return;
  const scenes = payload.per_scene || [];
  container.innerHTML = scenes
    .map((scene) => {
      const memberNames = payload.global?.member_names || [];
      const frameRows = (scene.frame_labels || [])
        .map((frame) => {
          const memberCols = memberNames
            .map((name) => `<td>${frame.member_clusters?.[name] ?? "—"}</td>`)
            .join("");
          return `<tr>
            <td>${frame.frame_number}</td>
            <td>${escapeHtml(frame.camera_id)}</td>
            <td>${frame.cluster_id}</td>
            <td>${frame.pre_merge_cluster_id ?? "—"}</td>
            ${memberCols}
          </tr>`;
        })
        .join("");
      const memberHeader = memberNames.map((name) => `<th>${escapeHtml(name)}</th>`).join("");
      const gt = scene.gt_camera ? `<div>GT camera: ${escapeHtml(scene.gt_camera)}</div>` : "";
      const mapped = scene.mapped_cluster_camera
        ? `<div>Hungarian-mapped eval label: ${escapeHtml(scene.mapped_cluster_camera)}</div>`
        : "";
      const preMerge = scene.scene_mode_cluster != null
        ? `<div>Pre-merge cluster (debug): ${scene.scene_mode_cluster}</div>`
        : "";

      return `<article class="compare-scene-card">
        <header>${escapeHtml(scene.camera_id)} · Scene ${scene.scene_id}</header>
        <img src="/api/scene-images/${scene.scene_id}/mid" alt="Scene ${scene.scene_id} mid frame" loading="lazy">
        <div><strong>Final scene assignment</strong> (vote + reconcile): ${escapeHtml(scene.winner)} · cluster ${scene.cluster_id}${scene.pred_noise ? " (noise)" : ""}</div>
        <div>Winner vote share: ${formatPct(scene.winner_share)}</div>
        ${preMerge}
        ${gt}
        ${mapped}
        ${voteBarHtml(scene.camera_vote_counts, scene.winner)}
        <p class="muted compare-frame-note">Frame table uses post-merge labels from <code>frame_camera_results.csv</code> (input to majority vote).</p>
        <div class="frame-label-table-wrap">
          <table class="frame-label-table">
            <thead><tr><th>Frame</th><th>Camera</th><th>Cluster</th><th>Pre-merge</th>${memberHeader}</tr></thead>
            <tbody>${frameRows}</tbody>
          </table>
        </div>
      </article>`;
    })
    .join("");
}

function renderGlobalLegend(projection) {
  const legendEl = document.getElementById("camera-global-legend");
  if (!legendEl) return;
  const legend = projection?.legend || [];
  if (!legend.length) {
    legendEl.innerHTML = "";
    return;
  }
  legendEl.innerHTML = legend
    .map((entry) => {
      const label =
        entry.cluster_id < 0
          ? "noise (-1)"
          : `cluster ${entry.cluster_id} → ${escapeHtml(entry.camera_id)}`;
      return `<span class="legend-item"><span class="legend-swatch" style="background:${entry.color}"></span>${label}</span>`;
    })
    .join("");
}

function renderGlobalCaption(projection) {
  const caption = document.getElementById("camera-global-caption");
  if (!caption || !projection) return;
  const method = projection.projection_method || "unknown";
  const featureMethod = projection.method || "ensemble";
  const labelSource = projection.label_source || "unknown";
  caption.textContent =
    `t-SNE on ${featureMethod} reduced features (${method}). ` +
    `Dot color = post-merge cluster_id from ${labelSource}. ` +
    `White rings = scenes on checked camera(s). Hover a point for assignment details. Scroll to zoom, drag to pan.`;
}

function drawProjectionPoint(ctx, point, screenX, screenY, { hovered = false } = {}) {
  const color = point.color || "#888888";
  const highlighted = Boolean(point.highlighted);
  const radius = highlighted ? 6 : hovered ? 5.5 : 3.5;

  ctx.beginPath();
  ctx.fillStyle = highlighted || hovered ? "#ffffff" : color;
  ctx.globalAlpha = highlighted || hovered ? 1 : 0.7;
  ctx.arc(screenX, screenY, radius, 0, Math.PI * 2);
  ctx.fill();

  if (highlighted || hovered) {
    ctx.strokeStyle = color;
    ctx.lineWidth = hovered && !highlighted ? 1.5 : 2;
    ctx.stroke();
  }

  if (highlighted) {
    ctx.fillStyle = "#ffffff";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`s${point.scene_id}`, screenX, screenY - 10);
  } else if (hovered) {
    ctx.fillStyle = "#ffffff";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`s${point.scene_id}`, screenX, screenY - 9);
  }
}

function refreshHoverFromPointer() {
  const wrap = document.getElementById("camera-global-canvas-wrap");
  const canvas = document.getElementById("camera-global-canvas");
  const pointer = globalProjectionState.lastPointer;
  if (!wrap || !canvas || !pointer || globalProjectionState.dragging) return;

  const anchor = canvasPointerCoords(canvas, pointer.clientX, pointer.clientY);
  const point = findPointAt(anchor.x, anchor.y);
  setHoveredPoint(point);
  if (point) {
    showGlobalTooltip(point, pointer.clientX, pointer.clientY, wrap);
    wrap.classList.add("hovering-point");
  } else {
    hideGlobalTooltip();
    wrap.classList.remove("hovering-point");
  }
}

function renderGlobalProjectionCanvas() {
  const canvas = document.getElementById("camera-global-canvas");
  const projection = globalProjectionState.projection;
  if (!canvas || !projection?.points?.length || !globalProjectionState.bounds) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const hoveredKey = pointKey(globalProjectionState.hoveredPoint);
  const points = projection.points;
  const screenPoints = points.map((point) => {
    const screen = dataToScreen(point.x, point.y, width, height);
    return { point, ...screen };
  });

  screenPoints.forEach(({ point, x, y }) => {
    if (pointKey(point) === hoveredKey) return;
    drawProjectionPoint(ctx, point, x, y);
  });

  const hovered = screenPoints.find(({ point }) => pointKey(point) === hoveredKey);
  if (hovered) {
    drawProjectionPoint(ctx, hovered.point, hovered.x, hovered.y, { hovered: true });
  }

  ctx.globalAlpha = 1;

  ctx.fillStyle = "#888";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("t-SNE dim 1 →", GLOBAL_PAD, height - 10);
  ctx.save();
  ctx.translate(12, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("t-SNE dim 2 →", 0, 0);
  ctx.restore();

  refreshHoverFromPointer();
}

function drawGlobalProjection(projection) {
  const canvas = document.getElementById("camera-global-canvas");
  if (!canvas || !projection?.points?.length) return;

  globalProjectionState.projection = projection;
  globalProjectionState.bounds = computeProjectionBounds(projection.points);
  globalProjectionState.hoveredPoint = null;
  hideGlobalTooltip();
  resetGlobalView();
  syncGlobalCanvasSize();

  renderGlobalCaption(projection);
  renderGlobalLegend(projection);
  renderGlobalProjectionCanvas();
}

async function runComparison(selections, includeGlobal) {
  const panel = document.getElementById("camera-compare-panel");
  const status = document.getElementById("camera-compare-status");
  if (!panel || !status) return;

  panel.classList.remove("hidden");
  panel.removeAttribute("hidden");
  status.textContent = "Comparing…";

  try {
    const res = await fetch("/api/cameras/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selections, include_global: includeGlobal }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || res.statusText);

    status.textContent = payload.has_debug_artifact
      ? `Compared ${selections.length} scene(s) using ${payload.global?.method || "clustering"} features.`
      : "Vote/cluster metadata only — re-run cameras stage for distance metrics.";

    renderExplanations(payload);
    renderPairwiseTable(payload);
    renderSceneCards(payload);

    if (includeGlobal && payload.global_projection) {
      drawGlobalProjection(payload.global_projection);
    } else if (includeGlobal) {
      const params = new URLSearchParams();
      selections.forEach((s) => params.append("camera_id", s.camera_id));
      const globalRes = await fetch(`/api/camera-debug/global?${params.toString()}`);
      const globalData = await globalRes.json();
      if (globalRes.ok) drawGlobalProjection(globalData);
    }
  } catch (err) {
    status.textContent = "";
    const explanations = document.getElementById("camera-compare-explanations");
    if (explanations) {
      explanations.innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
    } else {
      status.textContent = err.message;
      status.classList.add("error");
    }
  }
}

window.cameraCompare = {
  sceneLabel,
  runComparison,
  drawGlobalProjection,
};
