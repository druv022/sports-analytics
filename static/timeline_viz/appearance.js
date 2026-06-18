const appearanceState = {
  loaded: false,
  hasArtifacts: false,
  summary: {},
  scenes: [],
  compatibilityGroups: {},
  scenesWithIssues: [],
  selectedSceneId: null,
  sceneDetail: null,
  activeSlot: null,
  activeFrameNumber: null,
  view: {
    image: null,
    detections: [],
    storedFrame: null,
    zoom: 1,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
  },
  filters: {
    status: "",
    sceneType: "",
    personCount: "",
    varianceOnly: false,
  },
};

const APPEARANCE_MIN_ZOOM = 0.25;
const APPEARANCE_MAX_ZOOM = 8;

const COLOR_MAP = {
  red: "#e74c3c",
  orange: "#e67e22",
  yellow: "#f1c40f",
  green: "#2ecc71",
  blue: "#3498db",
  purple: "#9b59b6",
  white: "#ecf0f1",
  black: "#2c3e50",
  neutral: "#95a5a6",
};

let appearanceTabInitialized = false;

function appearanceEl(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function apiErrorMessage(data, fallback) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === "object" && item?.msg ? item.msg : String(item)))
      .join("; ");
  }
  if (detail != null) return String(detail);
  return fallback;
}

function colorSwatches(colors, primaryBgr = null) {
  if (primaryBgr && primaryBgr.length === 3) {
    const [b, g, r] = primaryBgr;
    const fill = `rgb(${r}, ${g}, ${b})`;
    const label = colors?.[0] || "primary";
    return `<span class="color-swatch" style="background:${fill}" title="${escapeHtml(label)} (BGR ${primaryBgr.join(",")})"></span>`;
  }
  if (!colors?.length) return '<span class="muted">—</span>';
  return colors
    .map((color) => {
      const fill = COLOR_MAP[color] || COLOR_MAP.neutral;
      return `<span class="color-swatch" style="background:${fill}" title="${escapeHtml(color)}"></span>`;
    })
    .join("");
}

function statusBadge(status) {
  const cls =
    status === "ok" ? "status-ok" : status === "low_conf" ? "status-warn" : "status-bad";
  return `<span class="appearance-status ${cls}">${escapeHtml(status)}</span>`;
}

function filteredScenes() {
  return appearanceState.scenes.filter((scene) => {
    if (appearanceState.filters.status && scene.status !== appearanceState.filters.status) {
      return false;
    }
    if (appearanceState.filters.sceneType && scene.scene_type !== appearanceState.filters.sceneType) {
      return false;
    }
    if (
      appearanceState.filters.personCount !== "" &&
      String(scene.person_count) !== appearanceState.filters.personCount
    ) {
      return false;
    }
    if (appearanceState.filters.varianceOnly && !scene.has_count_variance) {
      return false;
    }
    return true;
  });
}

function renderAppearanceBanner() {
  const banner = appearanceEl("appearance-banner");
  if (!banner) return;
  if (!appearanceState.hasArtifacts) {
    banner.classList.remove("hidden");
    banner.innerHTML =
      '<p class="error">Appearance artifacts not found. Run the pipeline through the <strong>appearance</strong> stage to generate <code>frame_appearance.csv</code> and <code>scene_appearance.csv</code>.</p>';
    return;
  }
  banner.classList.add("hidden");
  banner.innerHTML = "";
}

function renderAppearanceSummary() {
  const container = appearanceEl("appearance-summary");
  const histogram = appearanceEl("appearance-histogram");
  if (!container || !histogram) return;

  if (!appearanceState.hasArtifacts) {
    container.textContent = "No appearance data available.";
    histogram.innerHTML = "";
    return;
  }

  const s = appearanceState.summary;
  const nComponents = s.n_compatibility_components ?? 0;
  container.innerHTML = `
    <div class="appearance-summary-grid">
      <div class="summary-card"><strong>${s.n_scenes ?? 0}</strong><span>scenes</span></div>
      <div class="summary-card"><strong>${s.n_frames ?? 0}</strong><span>camera frames</span></div>
      <div class="summary-card"><strong>${nComponents}</strong><span>compat groups</span></div>
      <div class="summary-card"><strong>${s.n_scenes_with_count_variance ?? 0}</strong><span>count variance</span></div>
      <div class="summary-card"><strong>${s.n_scenes_with_issues ?? 0}</strong><span>issues</span></div>
    </div>
    <p class="muted appearance-status-line">
      Scene status: ${formatStatusCounts(s.scene_status_counts)}
      · Frame status: ${formatStatusCounts(s.frame_status_counts)}
    </p>
  `;

  const hist = s.person_count_histogram || {};
  const maxVal = Math.max(1, ...Object.values(hist));
  const bars = Object.entries(hist)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([count, value]) => {
      const height = Math.round((value / maxVal) * 100);
      return `<div class="hist-bar-wrap" title="${value} scene(s) with ${count} person(s)">
        <div class="hist-bar" style="height:${height}%"></div>
        <span class="hist-label">${escapeHtml(count)}</span>
        <span class="hist-value">${value}</span>
      </div>`;
    })
    .join("");
  histogram.innerHTML = bars
    ? `<h3 class="hist-title">Person count distribution (scenes)</h3><div class="hist-chart">${bars}</div>`
    : "";
}

function formatStatusCounts(counts) {
  if (!counts || !Object.keys(counts).length) return "—";
  return Object.entries(counts)
    .map(([key, val]) => `${key}: ${val}`)
    .join(", ");
}

function populateCountFilter() {
  const select = appearanceEl("appearance-filter-count");
  if (!select) return;
  const counts = new Set(appearanceState.scenes.map((s) => s.person_count));
  const current = appearanceState.filters.personCount;
  select.innerHTML = '<option value="">All</option>';
  [...counts]
    .sort((a, b) => a - b)
    .forEach((count) => {
      const opt = document.createElement("option");
      opt.value = String(count);
      opt.textContent = String(count);
      select.appendChild(opt);
    });
  select.value = current;
}

function renderAppearanceSceneTable() {
  const body = appearanceEl("appearance-scenes-body");
  const countLabel = appearanceEl("appearance-scene-count");
  if (!body) return;

  const rows = filteredScenes();
  if (countLabel) {
    countLabel.textContent = `${rows.length} of ${appearanceState.scenes.length} scene(s)`;
  }

  body.innerHTML = rows
    .map((scene) => {
      const selected = appearanceState.selectedSceneId === scene.scene_id ? "selected" : "";
      const variance = scene.has_count_variance
        ? '<span class="variance-flag" title="Frame counts disagree with scene majority">!</span>'
        : "";
      return `<tr class="appearance-scene-row ${selected}" data-scene-id="${scene.scene_id}">
        <td>${scene.scene_id}</td>
        <td>${escapeHtml(scene.camera_id || "—")}</td>
        <td>${escapeHtml(scene.scene_type)}</td>
        <td>${scene.person_count}</td>
        <td class="color-cell">${colorSwatches(scene.person_colors, scene.primary_bgr)}</td>
        <td class="mono">${escapeHtml(scene.appearance_signature)}</td>
        <td>${Number(scene.confidence).toFixed(2)}</td>
        <td>${statusBadge(scene.status)}</td>
        <td>${variance}</td>
      </tr>`;
    })
    .join("");

  body.querySelectorAll(".appearance-scene-row").forEach((row) => {
    row.addEventListener("click", () => {
      const sceneId = Number(row.dataset.sceneId);
      selectAppearanceScene(sceneId);
    });
  });
}

async function selectAppearanceScene(sceneId) {
  appearanceState.selectedSceneId = sceneId;
  renderAppearanceSceneTable();

  const inspector = appearanceEl("appearance-inspector");
  if (inspector) inspector.hidden = false;

  try {
    const res = await fetch(`/api/appearance/scene/${sceneId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(apiErrorMessage(data, res.statusText));
    appearanceState.sceneDetail = data;
    renderAppearanceInspector();
    const firstSlot = data.slots?.[0];
    if (firstSlot) {
      showAppearanceSlot(firstSlot.slot, firstSlot.frame_number, firstSlot.image_url);
    }
  } catch (err) {
    const meta = appearanceEl("appearance-inspector-meta");
    if (meta) meta.innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
  }
}

function renderAppearanceInspector() {
  const detail = appearanceState.sceneDetail;
  if (!detail) return;

  const meta = appearanceEl("appearance-inspector-meta");
  const slotStrip = appearanceEl("appearance-slot-strip");
  const framesBody = appearanceEl("appearance-frames-body");
  const scene = detail.scene;

  if (meta) {
    meta.innerHTML = `
      <p><strong>Scene ${scene.scene_id}</strong> · ${escapeHtml(scene.camera_id || "—")} · ${escapeHtml(scene.scene_type)}</p>
      <p>Signature: <span class="mono">${escapeHtml(scene.appearance_signature)}</span> · ${statusBadge(scene.status)}</p>
      <p>Colors: ${colorSwatches(scene.person_colors, scene.primary_bgr)} · confidence ${Number(scene.confidence).toFixed(2)}</p>
    `;
  }

  if (slotStrip) {
    slotStrip.innerHTML = (detail.slots || [])
      .map(
        (slot) => `<button type="button" class="appearance-slot-btn" data-slot="${escapeHtml(slot.slot)}" data-frame="${slot.frame_number}">
          ${escapeHtml(slot.slot)} #${slot.frame_number}
        </button>`,
      )
      .join("");
    slotStrip.querySelectorAll(".appearance-slot-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const slot = btn.dataset.slot;
        const frameNumber = Number(btn.dataset.frame);
        const imageUrl = `/api/scene-images/${scene.scene_id}/${slot}`;
        showAppearanceSlot(slot, frameNumber, imageUrl);
      });
    });
  }

  if (framesBody) {
    framesBody.innerHTML = (detail.frames || [])
      .map((frame) => {
        const active =
          appearanceState.activeFrameNumber === frame.frame_number ? "selected" : "";
        return `<tr class="appearance-frame-row ${active}" data-frame="${frame.frame_number}">
          <td>#${frame.frame_number}</td>
          <td>${frame.person_count}</td>
          <td class="color-cell">${colorSwatches(frame.person_colors, frame.primary_bgr)}</td>
          <td>${Number(frame.confidence).toFixed(2)}</td>
          <td>${statusBadge(frame.status)}</td>
        </tr>`;
      })
      .join("");
    framesBody.querySelectorAll(".appearance-frame-row").forEach((row) => {
      row.addEventListener("click", () => {
        const frameNumber = Number(row.dataset.frame);
        const slotMatch = (detail.slots || []).find((s) => s.frame_number === frameNumber);
        const imageUrl = slotMatch
          ? `/api/scene-images/${scene.scene_id}/${slotMatch.slot}`
          : `/api/frames/${frameNumber}`;
        showAppearanceSlot(slotMatch?.slot || "mid", frameNumber, imageUrl);
      });
    });
  }
}

async function showAppearanceSlot(slot, frameNumber, imageUrl) {
  appearanceState.activeSlot = slot;
  appearanceState.activeFrameNumber = frameNumber;

  document.querySelectorAll(".appearance-slot-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.slot === slot);
  });
  document.querySelectorAll(".appearance-frame-row").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.frame) === frameNumber);
  });

  const label = appearanceEl("appearance-frame-label");
  const status = appearanceEl("appearance-segment-status");
  if (label) label.textContent = `${slot} · frame ${frameNumber}`;
  if (status) status.textContent = "Loading…";

  const stored = (appearanceState.sceneDetail?.frames || []).find(
    (f) => f.frame_number === frameNumber,
  );
  appearanceState.view.storedFrame = stored || null;

  try {
    const img = await loadAppearanceImage(imageUrl);
    appearanceState.view.image = img;

    const liveEnabled = appearanceEl("appearance-live-segment")?.checked ?? true;
    if (liveEnabled) {
      const segRes = await fetch(
        `/api/appearance/segment/scene/${appearanceState.selectedSceneId}/${slot}`,
      );
      const payload = await segRes.json();
      if (!segRes.ok) throw new Error(apiErrorMessage(payload, segRes.statusText));
      appearanceState.view.detections = payload.detections || [];
      if (status) {
        const storedNote = stored
          ? `CSV: ${stored.person_count} person(s), ${stored.status}`
          : "";
        status.textContent = `Live: ${payload.person_count} person(s), ${payload.status}${storedNote ? ` · ${storedNote}` : ""}`;
      }
    } else {
      appearanceState.view.detections = [];
      if (status && stored) {
        status.textContent = `Stored CSV: ${stored.person_count} person(s), ${stored.status}`;
      } else if (status) {
        status.textContent = "Stored CSV only (no live overlay)";
      }
    }

    resetAppearanceZoom();
    renderAppearanceCanvas();
  } catch (err) {
    appearanceState.view.image = null;
    appearanceState.view.detections = [];
    clearAppearanceCanvas();
    if (status) {
      status.textContent = `Failed: ${err.message}`;
      status.classList.add("error");
    }
  }
}

function loadAppearanceImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load image: ${src}`));
    img.src = src;
  });
}

function clearAppearanceCanvas() {
  const canvas = appearanceEl("appearance-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function appearanceBaseScale(image) {
  const wrap = appearanceEl("appearance-canvas-wrap");
  const fitWidth = wrap?.clientWidth || 960;
  return fitWidth / image.width;
}

function resetAppearanceZoom() {
  appearanceState.view.zoom = 1;
  const wrap = appearanceEl("appearance-canvas-wrap");
  if (wrap) {
    wrap.scrollLeft = 0;
    wrap.scrollTop = 0;
  }
  updateAppearanceZoomLabel();
}

function setAppearanceZoom(next) {
  appearanceState.view.zoom = Math.min(
    APPEARANCE_MAX_ZOOM,
    Math.max(APPEARANCE_MIN_ZOOM, next),
  );
  updateAppearanceZoomLabel();
}

function updateAppearanceZoomLabel() {
  const label = appearanceEl("appearance-zoom-label");
  if (label) label.textContent = `${Math.round(appearanceState.view.zoom * 100)}%`;
}

function renderAppearanceCanvas() {
  const canvas = appearanceEl("appearance-canvas");
  const wrap = appearanceEl("appearance-canvas-wrap");
  const img = appearanceState.view.image;
  if (!canvas || !wrap || !img) {
    clearAppearanceCanvas();
    return;
  }

  const { zoom, detections } = appearanceState.view;
  const scale = appearanceBaseScale(img) * zoom;
  const displayW = Math.max(1, Math.round(img.width * scale));
  const displayH = Math.max(1, Math.round(img.height * scale));

  canvas.width = displayW;
  canvas.height = displayH;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, displayW, displayH);
  ctx.drawImage(img, 0, 0, displayW, displayH);

  detections.forEach((det) => {
    const color = COLOR_MAP[det.clothing_color] || COLOR_MAP.neutral;
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = color;
    (det.mask_contours || []).forEach((contour) => {
      if (!contour.length) return;
      ctx.beginPath();
      contour.forEach(([x, y], idx) => {
        const px = x * scale;
        const py = y * scale;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fill();
    });
    ctx.restore();

    const [x1, y1, x2, y2] = det.bbox;
    const bx1 = x1 * scale;
    const by1 = y1 * scale;
    const bw = (x2 - x1) * scale;
    const bh = (y2 - y1) * scale;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx1, by1, bw, bh);

    const label = `${det.clothing_color} ${(det.confidence * 100).toFixed(0)}%`;
    ctx.font = "12px system-ui, sans-serif";
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(bx1, Math.max(0, by1 - 18), tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, bx1 + 4, Math.max(12, by1 - 5));
  });
}

function setupAppearanceZoomControls() {
  const wrap = appearanceEl("appearance-canvas-wrap");
  appearanceEl("appearance-zoom-in")?.addEventListener("click", () => {
    setAppearanceZoom(appearanceState.view.zoom * 1.2);
    renderAppearanceCanvas();
  });
  appearanceEl("appearance-zoom-out")?.addEventListener("click", () => {
    setAppearanceZoom(appearanceState.view.zoom / 1.2);
    renderAppearanceCanvas();
  });
  appearanceEl("appearance-zoom-reset")?.addEventListener("click", () => {
    resetAppearanceZoom();
    renderAppearanceCanvas();
  });

  wrap?.addEventListener(
    "wheel",
    (event) => {
      if (!appearanceState.view.image) return;
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
      setAppearanceZoom(appearanceState.view.zoom * factor);
      renderAppearanceCanvas();
    },
    { passive: false },
  );

  wrap?.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || !appearanceState.view.image) return;
    appearanceState.view.dragging = true;
    appearanceState.view.dragStartX = event.clientX;
    appearanceState.view.dragStartY = event.clientY;
    wrap.classList.add("dragging");
  });

  window.addEventListener("mousemove", (event) => {
    if (!appearanceState.view.dragging || !wrap) return;
    wrap.scrollLeft -= event.clientX - appearanceState.view.dragStartX;
    wrap.scrollTop -= event.clientY - appearanceState.view.dragStartY;
    appearanceState.view.dragStartX = event.clientX;
    appearanceState.view.dragStartY = event.clientY;
  });

  window.addEventListener("mouseup", () => {
    if (!appearanceState.view.dragging) return;
    appearanceState.view.dragging = false;
    wrap?.classList.remove("dragging");
  });
}

function setupAppearanceFilters() {
  appearanceEl("appearance-filter-status")?.addEventListener("change", (event) => {
    appearanceState.filters.status = event.target.value;
    renderAppearanceSceneTable();
  });
  appearanceEl("appearance-filter-type")?.addEventListener("change", (event) => {
    appearanceState.filters.sceneType = event.target.value;
    renderAppearanceSceneTable();
  });
  appearanceEl("appearance-filter-count")?.addEventListener("change", (event) => {
    appearanceState.filters.personCount = event.target.value;
    renderAppearanceSceneTable();
  });
  appearanceEl("appearance-filter-variance")?.addEventListener("change", (event) => {
    appearanceState.filters.varianceOnly = event.target.checked;
    renderAppearanceSceneTable();
  });
  appearanceEl("appearance-live-segment")?.addEventListener("change", () => {
    if (appearanceState.activeSlot != null && appearanceState.activeFrameNumber != null) {
      const sceneId = appearanceState.selectedSceneId;
      const slot = appearanceState.activeSlot;
      const imageUrl = `/api/scene-images/${sceneId}/${slot}`;
      showAppearanceSlot(slot, appearanceState.activeFrameNumber, imageUrl);
    }
  });
}

async function loadAppearance() {
  const res = await fetch("/api/appearance");
  const data = await res.json();
  if (!res.ok) throw new Error(apiErrorMessage(data, res.statusText));

  appearanceState.hasArtifacts = Boolean(data.has_artifacts);
  appearanceState.summary = data.summary || {};
  appearanceState.compatibilityGroups = data.compatibility_groups || {};
  appearanceState.scenesWithIssues = data.scenes_with_issues || [];

  const scenesRes = await fetch("/api/appearance/scenes");
  const scenesData = await scenesRes.json();
  if (!scenesRes.ok) throw new Error(apiErrorMessage(scenesData, scenesRes.statusText));
  appearanceState.scenes = scenesData.scenes || [];
  appearanceState.loaded = true;

  renderAppearanceBanner();
  renderAppearanceSummary();
  populateCountFilter();
  renderAppearanceSceneTable();
}

function initAppearanceTab() {
  if (appearanceTabInitialized) return;
  appearanceTabInitialized = true;
  setupAppearanceZoomControls();
  setupAppearanceFilters();
}

window.initAppearanceTab = initAppearanceTab;
window.loadAppearance = loadAppearance;
