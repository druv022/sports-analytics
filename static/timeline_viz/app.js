const state = {
  rows: [],
  suggestions: [],
  summary: {},
  selectedKey: null,
  selectedRow: null,
  detailFrames: [],
  detailFrameDetails: [],
  activeFrame: null,
  ocrImage: null,
  suppressSuggestions: false,
  detailView: {
    image: null,
    detections: [],
    zoom: 1,
  },
  filters: {
    cameras: null,
    texts: null,
    kinds: null,
    mapped: null,
    durationMin: null,
    durationMax: null,
  },
  filterOptions: {
    cameras: [],
    texts: [],
    kinds: ["complete", "partial"],
    mapped: [],
  },
  filterDropdowns: {},
};

const DETAIL_MIN_ZOOM = 0.25;
const DETAIL_MAX_ZOOM = 8;

const els = {
  summary: document.getElementById("summary"),
  searchInput: document.getElementById("search-input"),
  suggestions: document.getElementById("suggestions"),
  resultCount: document.getElementById("result-count"),
  resultsBody: document.getElementById("results-body"),
  detailPanel: document.getElementById("detail-panel"),
  detailMeta: document.getElementById("detail-meta"),
  detailFrameLabel: document.getElementById("detail-frame-label"),
  detailOcrStatus: document.getElementById("detail-ocr-status"),
  detailFrameCanvas: document.getElementById("detail-frame-canvas"),
  detailCanvasWrap: document.getElementById("detail-canvas-wrap"),
  detailZoomOut: document.getElementById("detail-zoom-out"),
  detailZoomIn: document.getElementById("detail-zoom-in"),
  detailZoomReset: document.getElementById("detail-zoom-reset"),
  detailZoomLabel: document.getElementById("detail-zoom-label"),
  frameList: document.getElementById("frame-list"),
  ocrFile: document.getElementById("ocr-file"),
  ocrRun: document.getElementById("ocr-run"),
  ocrStatus: document.getElementById("ocr-status"),
  ocrCanvas: document.getElementById("ocr-canvas"),
  filterCamera: document.getElementById("filter-camera"),
  filterText: document.getElementById("filter-text"),
  filterKind: document.getElementById("filter-kind"),
  filterMapped: document.getElementById("filter-mapped"),
  filterDurationMin: document.getElementById("filter-duration-min"),
  filterDurationMax: document.getElementById("filter-duration-max"),
  filterReset: document.getElementById("filter-reset"),
};

function foldCase(text) {
  return String(text ?? "").toLocaleLowerCase();
}

function rowKey(row) {
  return `${row.camera_id}|${row.mapped_complete_text}|${row.text}|${row.text_kind}`;
}

function formatDurationSecMs(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  const secStr = num.toFixed(2);
  const ms = Math.round(num * 1000);
  return `${secStr}s · ${ms}ms`;
}

function formatDurationSeconds(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  const base = `${num.toFixed(2)}s`;
  if (num <= 60) return base;

  const total = Math.floor(num);
  const pad = (part) => String(part).padStart(2, "0");
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;

  if (hours > 0) {
    return `${base} (${hours}:${pad(minutes)}:${pad(seconds)})`;
  }
  return `${base} (${minutes}:${pad(seconds)})`;
}

function durationStats(rows) {
  const durations = rows
    .map((row) => Number(row.total_duration_sec))
    .filter(Number.isFinite);
  if (!durations.length) return null;
  return {
    min: Math.min(...durations),
    max: Math.max(...durations),
    total: durations.reduce((sum, value) => sum + value, 0),
    minFloor: Math.floor(Math.min(...durations)),
    maxFloor: Math.floor(Math.max(...durations)),
  };
}

function filterRowsExceptDuration(rows) {
  const savedMin = state.filters.durationMin;
  const savedMax = state.filters.durationMax;
  state.filters.durationMin = null;
  state.filters.durationMax = null;
  const result = filterRows(rows);
  state.filters.durationMin = savedMin;
  state.filters.durationMax = savedMax;
  return result;
}

function updateDurationFilterBounds() {
  if (!els.filterDurationMin || !els.filterDurationMax) return;
  const stats = durationStats(filterRowsExceptDuration(state.rows));
  if (!stats) {
    els.filterDurationMin.placeholder = "Min";
    els.filterDurationMax.placeholder = "Max";
    return;
  }
  els.filterDurationMin.placeholder = String(stats.minFloor);
  els.filterDurationMax.placeholder = String(stats.maxFloor);
}

function uniqueSorted(values) {
  const seen = new Set();
  const ordered = [];
  for (const value of values) {
    const token = String(value ?? "").trim();
    if (!token || seen.has(token)) continue;
    seen.add(token);
    ordered.push(token);
  }
  return ordered.sort((a, b) => a.localeCompare(b));
}

function buildFilterOptions(rows) {
  state.filterOptions = {
    cameras: uniqueSorted(rows.map((row) => row.camera_id)),
    texts: uniqueSorted(rows.map((row) => row.text)),
    kinds: uniqueSorted(
      rows.map((row) => row.text_kind).filter((kind) => kind === "complete" || kind === "partial"),
    ),
    mapped: uniqueSorted(rows.map((row) => row.mapped_complete_text)),
  };
  if (!state.filterOptions.kinds.length) {
    state.filterOptions.kinds = ["complete", "partial"];
  }
}

function resetFilters() {
  state.filters = {
    cameras: null,
    texts: null,
    kinds: null,
    mapped: null,
    durationMin: null,
    durationMax: null,
  };
  if (els.filterDurationMin) els.filterDurationMin.value = "";
  if (els.filterDurationMax) els.filterDurationMax.value = "";
  refreshFilterDropdowns();
}

function isFilterActive() {
  const f = state.filters;
  const opts = state.filterOptions;
  const dimensionActive = (selected, options) =>
    selected !== null && (selected.size === 0 || selected.size < options.length);
  return (
    dimensionActive(f.cameras, opts.cameras) ||
    dimensionActive(f.texts, opts.texts) ||
    dimensionActive(f.kinds, opts.kinds) ||
    dimensionActive(f.mapped, opts.mapped) ||
    f.durationMin != null ||
    f.durationMax != null
  );
}

function matchesFilterSet(selected, value) {
  if (selected === null) return true;
  if (selected.size === 0) return false;
  return selected.has(String(value ?? ""));
}

function filterRows(rows) {
  const f = state.filters;
  const query = els.searchInput?.value.trim() ?? "";
  const needle = foldCase(query);

  return rows.filter((row) => {
    if (!matchesFilterSet(f.cameras, row.camera_id)) return false;
    if (!matchesFilterSet(f.texts, row.text)) return false;
    if (!matchesFilterSet(f.kinds, row.text_kind)) return false;
    if (!matchesFilterSet(f.mapped, row.mapped_complete_text)) return false;

    if (needle && !foldCase(row.mapped_complete_text).includes(needle)) return false;

    const durationSec = Number(row.total_duration_sec);
    if (Number.isFinite(durationSec)) {
      const floored = Math.floor(durationSec);
      if (f.durationMin != null && floored < f.durationMin) return false;
      if (f.durationMax != null && floored > f.durationMax) return false;
    } else if (f.durationMin != null || f.durationMax != null) {
      return false;
    }

    return true;
  });
}

function searchSuggestions(query, limit = 20) {
  if (!query.trim()) return state.suggestions.slice(0, limit);
  const needle = foldCase(query);

  function rank(text) {
    const folded = foldCase(text);
    if (folded.startsWith(needle)) return 0;
    if (folded.includes(needle)) return 1;
    return 2;
  }

  return state.suggestions
    .filter((text) => foldCase(text).includes(needle))
    .sort((a, b) => {
      const ra = rank(a);
      const rb = rank(b);
      if (ra !== rb) return ra - rb;
      return a.localeCompare(b);
    })
    .slice(0, limit);
}

function closeAllFilterDropdowns(except = null) {
  Object.values(state.filterDropdowns).forEach((dropdown) => {
    if (dropdown !== except) dropdown.classList.remove("open");
  });
}

function filterTriggerLabel(label, selected, options) {
  if (selected === null || selected.size === options.length) {
    return `${label} (all)`;
  }
  if (selected.size === 0) {
    return `${label} (none)`;
  }
  return `${label} (${selected.size}/${options.length})`;
}

function isDropdownActive(selected, options) {
  return selected !== null && (selected.size === 0 || selected.size < options.length);
}

function createFilterDropdown(container, key, label, options, onChange) {
  if (!container) return null;
  container.innerHTML = "";
  const root = document.createElement("div");
  root.className = "filter-dropdown";
  root.dataset.filterKey = key;

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "filter-dropdown-trigger";
  trigger.setAttribute("aria-expanded", "false");
  trigger.setAttribute("aria-haspopup", "listbox");

  const panel = document.createElement("div");
  panel.className = "filter-dropdown-panel";
  panel.setAttribute("role", "listbox");

  const actions = document.createElement("div");
  actions.className = "filter-dropdown-actions";

  const selectAllBtn = document.createElement("button");
  selectAllBtn.type = "button";
  selectAllBtn.textContent = "Select all";

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.textContent = "Clear";

  actions.append(selectAllBtn, clearBtn);
  panel.appendChild(actions);

  const optionList = document.createElement("div");
  panel.appendChild(optionList);

  function getSelected() {
    return state.filters[key];
  }

  function setSelected(next) {
    state.filters[key] = next;
    syncCheckboxes();
    updateTrigger();
    onChange();
  }

  function syncCheckboxes() {
    const selected = getSelected();
    optionList.querySelectorAll("input[type=checkbox]").forEach((input) => {
      input.checked = selected === null || selected.has(input.value);
    });
  }

  function updateTrigger() {
    const selected = getSelected();
    trigger.textContent = filterTriggerLabel(label, selected, options);
    trigger.classList.toggle("filter-active", isDropdownActive(selected, options));
  }

  options.forEach((option) => {
    const row = document.createElement("label");
    row.className = "filter-dropdown-option";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = option;

    const text = document.createElement("span");
    text.textContent = option;

    input.addEventListener("change", () => {
      let selected = getSelected();
      if (selected === null) {
        selected = new Set(options);
      } else {
        selected = new Set(selected);
      }

      if (input.checked) {
        selected.add(option);
      } else {
        selected.delete(option);
      }

      if (selected.size === options.length) {
        setSelected(null);
      } else {
        setSelected(selected);
      }
    });

    row.append(input, text);
    optionList.appendChild(row);
  });

  selectAllBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    setSelected(null);
  });

  clearBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    setSelected(new Set());
  });

  trigger.addEventListener("click", (event) => {
    event.stopPropagation();
    const wasOpen = root.classList.contains("open");
    closeAllFilterDropdowns();
    if (!wasOpen) {
      root.classList.add("open");
      trigger.setAttribute("aria-expanded", "true");
    } else {
      trigger.setAttribute("aria-expanded", "false");
    }
  });

  panel.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  syncCheckboxes();
  updateTrigger();

  root.append(trigger, panel);
  container.appendChild(root);
  state.filterDropdowns[key] = root;
  return root;
}

function renderFilterDropdowns() {
  state.filterDropdowns = {};
  const opts = state.filterOptions;

  createFilterDropdown(els.filterCamera, "cameras", "Camera", opts.cameras, () => applyView());
  createFilterDropdown(els.filterText, "texts", "Text", opts.texts, () => applyView());
  createFilterDropdown(els.filterKind, "kinds", "Kind", opts.kinds, () => applyView());
  createFilterDropdown(els.filterMapped, "mapped", "Mapped", opts.mapped, () => applyView());
}

function refreshFilterDropdowns() {
  renderFilterDropdowns();
}

function parseDurationInput(input) {
  const raw = input.value.trim();
  if (!raw) return null;
  const num = Number.parseInt(raw, 10);
  return Number.isFinite(num) && num >= 0 ? num : null;
}

function syncDurationFilters() {
  state.filters.durationMin = parseDurationInput(els.filterDurationMin);
  state.filters.durationMax = parseDurationInput(els.filterDurationMax);
}

function clearSelectedRow() {
  state.selectedKey = null;
  state.selectedRow = null;
  if (els.detailPanel) els.detailPanel.hidden = true;
}

async function applyView(options = {}) {
  const { autoSelect = false } = options;
  els.resultCount.classList.remove("error");

  try {
    syncDurationFilters();
    updateDurationFilterBounds();
    const filtered = filterRows(state.rows);
    renderTable(filtered);

    const query = els.searchInput.value;
    if (!state.suppressSuggestions) {
      renderSuggestions(searchSuggestions(query), Boolean(query.trim()));
    } else {
      renderSuggestions([], false);
    }

    if (state.selectedKey) {
      const stillVisible = filtered.some((row) => rowKey(row) === state.selectedKey);
      if (!stillVisible) {
        clearSelectedRow();
      }
    }

    if (autoSelect && filtered.length) {
      const exact = filtered.find(
        (row) => foldCase(row.mapped_complete_text) === foldCase(query),
      );
      await selectRow(exact || filtered[0], rowKey(exact || filtered[0]));
    }
  } catch (err) {
    els.resultCount.textContent = `Filter failed: ${err.message}`;
    els.resultCount.classList.add("error");
  }
}

function renderSummary() {
  const s = state.summary || {};
  const parts = [];
  if (s.video_path) parts.push(`Video: ${s.video_path.split("/").pop()}`);
  if (s.duration_sec) parts.push(`Duration: ${formatDurationSecMs(s.duration_sec)}`);
  if (s.n_cameras) parts.push(`Cameras: ${s.n_cameras}`);
  if (s.n_ocr_frames) parts.push(`OCR frames: ${s.n_ocr_frames}`);
  if (s.output_dir) parts.push(`Output: ${s.output_dir}`);
  els.summary.textContent = parts.length ? parts.join(" · ") : "No pipeline summary available.";
}

function formatEnrichedCell(row) {
  const present = Number(row.n_frames_present);
  const enriched = Number(row.n_frames_enriched);
  if (!Number.isFinite(present) || present <= 0) return "—";
  if (!Number.isFinite(enriched) || enriched <= 0) return `0/${present}`;
  return `${enriched}/${present}`;
}

function formatFrameProvenanceLabel(frame) {
  if (!frame?.enrich_applied) return null;
  const from = frame.ocr_raw_text || "—";
  const to = frame.associated_text || state.selectedRow?.text || "";
  return `${from} → ${to}`;
}

function renderDurationSummaryRow(label, durationText, className) {
  const tr = document.createElement("tr");
  tr.className = className;
  tr.innerHTML = `
    <td colspan="5">${escapeHtml(label)}</td>
    <td>${escapeHtml(durationText)}</td>
    <td></td>
  `;
  return tr;
}

function renderTable(rows) {
  els.resultsBody.innerHTML = "";
  const stats = durationStats(rows);
  const showSummary = Boolean(stats) && (isFilterActive() || els.searchInput.value.trim());

  if (showSummary) {
    els.resultsBody.appendChild(
      renderDurationSummaryRow(
        "Aggregated total",
        formatDurationSeconds(stats.total),
        "duration-summary duration-summary-top",
      ),
    );
  }

  const sorted = [...rows].sort(
    (a, b) => Number(b.total_duration_sec) - Number(a.total_duration_sec),
  );

  sorted.forEach((row) => {
    const tr = document.createElement("tr");
    tr.dataset.key = rowKey(row);
    if (tr.dataset.key === state.selectedKey) tr.classList.add("selected");

    const kind = row.text_kind || "unknown";
    const enriched = formatEnrichedCell(row);
    const enrichedClass = enriched.includes("/") && !enriched.startsWith("0/") ? "badge enriched" : "";
    tr.innerHTML = `
      <td>${escapeHtml(row.camera_id || "")}</td>
      <td>${escapeHtml(row.text || "")}</td>
      <td><span class="badge ${kind}">${escapeHtml(kind)}</span></td>
      <td>${enrichedClass ? `<span class="${enrichedClass}">${escapeHtml(enriched)}</span>` : escapeHtml(enriched)}</td>
      <td>${escapeHtml(row.mapped_complete_text || "")}</td>
      <td>${formatDurationSeconds(row.total_duration_sec)}</td>
      <td><code>${escapeHtml(row.frame_ranges || "")}</code></td>
    `;
    tr.addEventListener("click", () => selectRow(row, tr.dataset.key));
    els.resultsBody.appendChild(tr);
  });

  if (showSummary) {
    els.resultsBody.appendChild(
      renderDurationSummaryRow(
        "Duration range",
        `${formatDurationSeconds(stats.min)} – ${formatDurationSeconds(stats.max)}`,
        "duration-summary duration-summary-bottom",
      ),
    );
  }

  const total = state.rows.length;
  const suffix = rows.length === 1 ? "" : "s";
  const countParts = [];
  if (isFilterActive() || els.searchInput.value.trim()) {
    countParts.push(`${rows.length} row${suffix} (filtered from ${total})`);
  } else {
    countParts.push(`${rows.length} row${suffix}`);
  }
  if (showSummary) {
    countParts.push(`aggregated ${formatDurationSeconds(stats.total)}`);
  }
  els.resultCount.textContent = countParts.join(" · ");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function fetchRowDetail(row) {
  const params = new URLSearchParams({
    camera_id: row.camera_id,
    mapped: row.mapped_complete_text || "",
    text: row.text || "",
  });
  const res = await fetch(`/api/row?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function renderSuggestions(items, forceShow = true) {
  els.suggestions.innerHTML = "";
  if (!items.length || !forceShow) {
    els.suggestions.classList.add("hidden");
    return;
  }
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    li.setAttribute("role", "option");
    li.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      selectSuggestion(text);
    });
    els.suggestions.appendChild(li);
  });
  els.suggestions.classList.remove("hidden");
}

function selectSuggestion(text) {
  state.suppressSuggestions = true;
  els.searchInput.value = text;
  els.suggestions.classList.add("hidden");
  applyView({ autoSelect: true }).finally(() => {
    state.suppressSuggestions = false;
  });
}

async function selectRow(row, key) {
  state.selectedKey = key;
  state.selectedRow = row;
  document.querySelectorAll("#results-body tr").forEach((tr) => {
    tr.classList.toggle("selected", tr.dataset.key === key);
  });

  els.detailPanel.hidden = false;
  const enrichedPresent = Number(row.n_frames_present);
  const enrichedCount = Number(row.n_frames_enriched);
  const enrichedMeta =
    Number.isFinite(enrichedPresent) && enrichedPresent > 0
      ? `${Number.isFinite(enrichedCount) ? enrichedCount : 0} / ${enrichedPresent}`
      : "—";
  els.detailMeta.innerHTML = `
    <div><dt>Camera ID</dt><dd>${escapeHtml(row.camera_id || "")}</dd></div>
    <div><dt>Text</dt><dd>${escapeHtml(row.text || "")}</dd></div>
    <div><dt>Kind</dt><dd>${escapeHtml(row.text_kind || "")}</dd></div>
    <div><dt>Mapped complete text</dt><dd>${escapeHtml(row.mapped_complete_text || "")}</dd></div>
    <div><dt>Enriched frames</dt><dd>${escapeHtml(enrichedMeta)}</dd></div>
    <div><dt>Total duration</dt><dd>${formatDurationSecMs(row.total_duration_sec)}</dd></div>
    <div><dt>Frame ranges</dt><dd><code>${escapeHtml(row.frame_ranges || "")}</code></dd></div>
  `;

  try {
    const detail = await fetchRowDetail(row);
    renderFrameDetail(detail);
  } catch (err) {
    els.frameList.innerHTML = `<li class="error">${escapeHtml(err.message)}</li>`;
    els.detailOcrStatus.textContent = "";
    clearCanvas(els.detailFrameCanvas);
  }
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  canvas.width = 0;
  canvas.height = 0;
  ctx.clearRect(0, 0, 0, 0);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function detectionMatchesText(det, text) {
  if (!text) return false;
  const detText = foldCase(det.text);
  const target = foldCase(text);
  if (detText === target) return true;
  if (detText.includes(target) || target.includes(detText)) return true;
  const detWords = detText.split(/\s+/);
  const targetWords = target.split(/\s+/);
  return detWords.some((word) =>
    targetWords.some((part) => word.includes(part) || part.includes(word)),
  );
}

function detectionMatchesRow(det, row) {
  if (!row) return false;
  const targets = [row.text, row.mapped_complete_text].filter(Boolean);
  if (!targets.length) return false;
  return targets.some((target) => detectionMatchesText(det, target));
}

function filterDetectionsForRow(detections, row) {
  if (!row || !detections?.length) return [];
  return detections.filter((det) => detectionMatchesRow(det, row));
}

function filterDetectionsForText(detections, text) {
  if (!text || !detections?.length) return [];
  return detections.filter((det) => detectionMatchesText(det, text));
}

function annotateEnrichedDetections(detections, associatedText) {
  const to = associatedText || "";
  return detections.map((det) => ({
    ...det,
    displayText: to ? `${det.text} → ${to}` : det.text,
  }));
}

function resetDetailZoom() {
  state.detailView.zoom = 1;
  if (els.detailCanvasWrap) {
    els.detailCanvasWrap.scrollLeft = 0;
    els.detailCanvasWrap.scrollTop = 0;
  }
  updateDetailZoomLabel();
}

function updateDetailZoomLabel() {
  if (els.detailZoomLabel) {
    els.detailZoomLabel.textContent = `${Math.round(state.detailView.zoom * 100)}%`;
  }
}

function detailBaseScale(image) {
  const fitWidth = els.detailCanvasWrap?.clientWidth || 960;
  return fitWidth / image.width;
}

function renderDetailCanvas() {
  const { image, detections, zoom } = state.detailView;
  if (!image) return;

  const canvas = els.detailFrameCanvas;
  const ctx = canvas.getContext("2d");
  const scale = detailBaseScale(image) * zoom;

  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  if (detections?.length) {
    drawOcrOverlay(ctx, detections, scale, { showConfidence: true });
  }
}

function setDetailZoom(nextZoom) {
  state.detailView.zoom = clamp(nextZoom, DETAIL_MIN_ZOOM, DETAIL_MAX_ZOOM);
  updateDetailZoomLabel();
  renderDetailCanvas();
}

function setupDetailZoomControls() {
  if (!els.detailCanvasWrap) return;

  els.detailCanvasWrap.addEventListener(
    "wheel",
    (event) => {
      if (!state.detailView.image) return;
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      setDetailZoom(state.detailView.zoom * factor);
    },
    { passive: false },
  );

  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  els.detailCanvasWrap.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || !state.detailView.image) return;
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    els.detailCanvasWrap.classList.add("dragging");
  });

  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    els.detailCanvasWrap.scrollLeft -= event.clientX - lastX;
    els.detailCanvasWrap.scrollTop -= event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
    els.detailCanvasWrap.classList.remove("dragging");
  });

  els.detailZoomIn?.addEventListener("click", () => {
    setDetailZoom(state.detailView.zoom * 1.2);
  });
  els.detailZoomOut?.addEventListener("click", () => {
    setDetailZoom(state.detailView.zoom / 1.2);
  });
  els.detailZoomReset?.addEventListener("click", () => {
    resetDetailZoom();
    renderDetailCanvas();
  });
}

function renderFrameDetail(detail) {
  state.detailFrames = detail.frames || [];
  state.detailFrameDetails = detail.frame_details || [];
  els.frameList.innerHTML = "";

  if (!state.detailFrames.length) {
    els.detailFrameLabel.textContent = "No frames available";
    els.detailOcrStatus.textContent = "";
    clearCanvas(els.detailFrameCanvas);
    return;
  }

  state.detailFrameDetails.forEach((frame) => {
    const li = document.createElement("li");
    const parts = [`#${frame.frame_number}`];
    if (frame.seconds != null) parts.push(formatDurationSecMs(frame.seconds));
    const prov = formatFrameProvenanceLabel(frame);
    if (prov) parts.push(prov);
    else if (frame.camera_id) parts.push(frame.camera_id);
    li.textContent = parts.join(" · ");
    li.dataset.frame = String(frame.frame_number);
    li.addEventListener("click", () => showFramePreview(frame.frame_number));
    els.frameList.appendChild(li);
  });

  const preferred = state.detailFrames.find((n) =>
    state.detailFrameDetails.some((f) => f.has_image && f.frame_number === n),
  );
  showFramePreview(preferred ?? state.detailFrames[0]);
}

function frameDetailFor(frameNumber) {
  return state.detailFrameDetails.find((f) => f.frame_number === frameNumber);
}

function setActiveFrameChip(frameNumber) {
  document.querySelectorAll("#frame-list li").forEach((li) => {
    li.classList.toggle("active", li.dataset.frame === String(frameNumber));
  });
}

async function showFramePreview(frameNumber) {
  state.activeFrame = frameNumber;
  setActiveFrameChip(frameNumber);
  els.detailFrameLabel.textContent = `Frame ${frameNumber}`;
  els.detailOcrStatus.textContent = "Loading image…";

  try {
    const img = await loadImage(`/api/frames/${frameNumber}`);
    els.detailOcrStatus.textContent = "Running OCR…";

    const ocrRes = await fetch(`/api/ocr/frame/${frameNumber}`);
    const payload = await ocrRes.json();
    if (!ocrRes.ok) throw new Error(payload.detail || ocrRes.statusText);

    const label = state.selectedRow?.text || state.selectedRow?.mapped_complete_text || "text";
    const frameDetail = frameDetailFor(frameNumber);
    let matched = filterDetectionsForRow(payload.detections || [], state.selectedRow);
    let enrichedHighlight = false;

    if (
      !matched.length &&
      frameDetail?.enrich_applied &&
      frameDetail.ocr_raw_text
    ) {
      const rawMatches = filterDetectionsForText(
        payload.detections || [],
        frameDetail.ocr_raw_text,
      );
      if (rawMatches.length) {
        matched = annotateEnrichedDetections(
          rawMatches,
          frameDetail.associated_text || label,
        );
        enrichedHighlight = true;
      }
    }

    state.detailView.image = img;
    state.detailView.detections = matched;
    resetDetailZoom();
    renderDetailCanvas();

    if (matched.length && enrichedHighlight) {
      const from = frameDetail.ocr_raw_text || "—";
      const to = frameDetail.associated_text || label;
      els.detailOcrStatus.textContent =
        `Highlighted "${from} → ${to}" (enriched) · Live OCR: no ${label} match`;
    } else if (matched.length) {
      els.detailOcrStatus.textContent = `Highlighted "${label}" · ${matched.length} match(es)`;
    } else if (frameDetail?.enrich_applied) {
      const from = frameDetail.ocr_raw_text || "—";
      const to = frameDetail.associated_text || label;
      els.detailOcrStatus.textContent =
        `Pipeline: ${from} → ${to} (enriched) · Live OCR: no ${label} match`;
    } else {
      els.detailOcrStatus.textContent = `No OCR match for "${label}" on this frame`;
    }
    els.detailOcrStatus.classList.remove("error");
  } catch (err) {
    state.detailView.image = null;
    state.detailView.detections = [];
    clearCanvas(els.detailFrameCanvas);
    els.detailOcrStatus.textContent = `Preview failed: ${err.message}`;
    els.detailOcrStatus.classList.add("error");
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load image: ${src}`));
    img.src = src;
  });
}

function labelPosition(bbox) {
  const [x0, y0, , y1] = bbox;
  const labelH = 18;
  if (y0 - labelH - 2 >= 0) {
    return { x: x0, y: y0 - labelH - 2, above: true };
  }
  return { x: x0, y: y1 + 2, above: false };
}

function confidenceColor(confidence) {
  if (confidence >= 0.85) return "#3ecf8e";
  if (confidence >= 0.6) return "#5b9dff";
  return "#f0b429";
}

function drawOcrOverlay(ctx, detections, scale, options = {}) {
  const { showConfidence = true } = options;
  detections.forEach((det) => {
    const [x0, y0, x1, y1] = det.bbox.map((v) => v * scale);
    const color = confidenceColor(det.confidence);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.22;
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.globalAlpha = 1;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    const labelText = det.displayText ?? det.text;
    const label = showConfidence
      ? `${labelText} (${det.confidence.toFixed(2)})`
      : labelText;
    ctx.font = "14px system-ui, sans-serif";
    const metrics = ctx.measureText(label);
    const pad = 4;
    const pos = labelPosition(det.bbox);
    const lx = pos.x * scale;
    const ly = pos.y * scale;
    const boxH = 18;
    ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
    ctx.fillRect(lx, ly, metrics.width + pad * 2, boxH);
    ctx.fillStyle = color;
    ctx.fillText(label, lx + pad, ly + boxH - 5);
  });
}

function drawOcrImage(canvas, image, detections, options = {}) {
  const ctx = canvas.getContext("2d");
  const wrap = canvas.parentElement;
  const maxWidth = options.maxWidth || wrap?.clientWidth || 960;
  const scale = Math.min(1, maxWidth / image.width);
  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (detections && detections.length) {
    drawOcrOverlay(ctx, detections, scale);
  }
}

async function runOcr() {
  if (!state.ocrImage) return;
  els.ocrStatus.textContent = "Running OCR…";
  els.ocrRun.disabled = true;

  const form = new FormData();
  form.append("file", state.ocrImage);

  try {
    const res = await fetch("/api/ocr", { method: "POST", body: form });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || res.statusText);

    const img = await loadImage(URL.createObjectURL(state.ocrImage));
    drawOcrImage(els.ocrCanvas, img, payload.detections || []);

    els.ocrStatus.textContent = `${payload.verdict} · ${payload.detections.length} detection(s)`;
    els.ocrStatus.classList.remove("error");
  } catch (err) {
    els.ocrStatus.textContent = `OCR failed: ${err.message}`;
    els.ocrStatus.classList.add("error");
  } finally {
    els.ocrRun.disabled = false;
  }
}

function setupFilterControls() {
  let durationDebounce = null;

  els.filterDurationMin?.addEventListener("input", () => {
    clearTimeout(durationDebounce);
    durationDebounce = setTimeout(() => applyView(), 200);
  });

  els.filterDurationMax?.addEventListener("input", () => {
    clearTimeout(durationDebounce);
    durationDebounce = setTimeout(() => applyView(), 200);
  });

  els.filterReset?.addEventListener("click", () => {
    resetFilters();
    applyView();
  });
}

async function init() {
  initTabs();
  window.initCamerasTab?.();
  window.initAppearanceTab?.();

  try {
    const res = await fetch("/api/timeline");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.rows = data.rows || [];
    state.suggestions = data.suggestions || [];
    state.summary = data.summary || {};
    renderSummary();
    buildFilterOptions(state.rows);
    renderFilterDropdowns();
    applyView();
  } catch (err) {
    els.summary.textContent = `Failed to load timeline: ${err.message}`;
    els.summary.classList.add("error");
  }

  setupFilterControls();

  let debounce = null;
  els.searchInput.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => applyView(), 180);
  });

  els.searchInput.addEventListener("focus", () => {
    if (state.suppressSuggestions) return;
    const query = els.searchInput.value.trim();
    if (query) {
      renderSuggestions(searchSuggestions(query), true);
    } else {
      renderSuggestions(state.suggestions.slice(0, 20));
    }
  });

  els.searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      const query = els.searchInput.value.trim();
      els.suggestions.classList.add("hidden");
      applyView({ autoSelect: true });
    } else if (event.key === "Escape") {
      els.suggestions.classList.add("hidden");
      closeAllFilterDropdowns();
    }
  });

  document.addEventListener("click", (event) => {
    if (
      !els.searchInput.contains(event.target) &&
      !els.suggestions.contains(event.target)
    ) {
      els.suggestions.classList.add("hidden");
    }

    const insideDropdown = Object.values(state.filterDropdowns).some((dropdown) =>
      dropdown.contains(event.target),
    );
    if (!insideDropdown) {
      closeAllFilterDropdowns();
      Object.values(state.filterDropdowns).forEach((dropdown) => {
        const trigger = dropdown.querySelector(".filter-dropdown-trigger");
        trigger?.setAttribute("aria-expanded", "false");
      });
    }
  });

  els.suggestions.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  els.ocrFile.addEventListener("change", () => {
    const file = els.ocrFile.files?.[0];
    state.ocrImage = file || null;
    els.ocrRun.disabled = !file;
    els.ocrStatus.textContent = file ? file.name : "";
    if (file) {
      loadImage(URL.createObjectURL(file)).then((img) => {
        drawOcrImage(els.ocrCanvas, img, null);
      });
    }
  });

  els.ocrRun.addEventListener("click", runOcr);

  setupDetailZoomControls();
  updateDetailZoomLabel();
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab-bar .tab");
  const panels = {
    timeline: document.getElementById("tab-timeline"),
    cameras: document.getElementById("tab-cameras"),
    appearance: document.getElementById("tab-appearance"),
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
      const name = tab.dataset.tab;
      tabs.forEach((btn) => {
        const active = btn === tab;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
      });
      Object.entries(panels).forEach(([key, panel]) => {
        if (!panel) return;
        const show = key === name;
        panel.hidden = !show;
      });
      if (name === "cameras") {
        window.initCamerasTab?.();
        if (typeof window.loadCameras === "function") {
          try {
            await window.loadCameras();
          } catch (_err) {
            // loadCameras reports errors in the camera collage panel.
          }
        }
      }
      if (name === "appearance") {
        window.initAppearanceTab?.();
        if (typeof window.loadAppearance === "function") {
          try {
            await window.loadAppearance();
          } catch (err) {
            const summary = document.getElementById("appearance-summary");
            if (summary) {
              summary.textContent = `Failed to load appearance data: ${err.message}`;
              summary.classList.add("error");
            }
          }
        }
      }
    });
  });
}

function boot() {
  init();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
