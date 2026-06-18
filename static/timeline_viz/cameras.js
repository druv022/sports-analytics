const cameraState = {
  loaded: false,
  cameraIds: [],
  counts: {},
  clusterIds: {},
  hasDebugArtifact: false,
  scenesByCamera: {},
  selectedCameras: new Set(),
  selectedScenes: {},
};

let camerasTabInitialized = false;

function cameraEl(id) {
  return document.getElementById(id);
}

function showCameraError(message) {
  const text = `<p class="error">${escapeHtml(message)}</p>`;
  const collage = cameraEl("camera-collage");
  if (collage) {
    collage.innerHTML = text;
    return;
  }
  const sceneCount = cameraEl("camera-scene-count");
  if (sceneCount) {
    sceneCount.textContent = message;
    sceneCount.classList.add("error");
  }
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

function formatSceneRange(startSec, endSec) {
  if (startSec == null || endSec == null) return "";
  return `${Number(startSec).toFixed(1)}s – ${Number(endSec).toFixed(1)}s`;
}

function filteredCameraIds() {
  const query = (cameraEl("camera-filter")?.value || "").trim().toLowerCase();
  if (!query) return cameraState.cameraIds;
  return cameraState.cameraIds.filter((id) => id.toLowerCase().includes(query));
}

function renderCameraCheckboxes() {
  const checkboxList = cameraEl("camera-checkbox-list");
  if (!checkboxList) {
    showCameraError(
      "Camera picker failed to load. Hard-refresh this page (Chrome: Ctrl+Shift+R, Mac: Cmd+Shift+R).",
    );
    return;
  }
  const ids = filteredCameraIds();
  checkboxList.innerHTML = ids
    .map((cameraId) => {
      const count = cameraState.counts[cameraId] ?? 0;
      const clusterId = cameraState.clusterIds[cameraId];
      const noiseBadge = clusterId === -1 ? '<span class="badge-noise">noise</span>' : "";
      const checked = cameraState.selectedCameras.has(cameraId) ? "checked" : "";
      return `<label class="camera-checkbox-item">
        <input type="checkbox" value="${escapeHtml(cameraId)}" ${checked}>
        <span>${escapeHtml(cameraId)} (${count})</span>
        ${noiseBadge}
      </label>`;
    })
    .join("");

  checkboxList.querySelectorAll("input[type=checkbox]").forEach((input) => {
    input.addEventListener("change", () => onCameraSelectionChanged());
  });
}

async function ensureScenesForCamera(cameraId) {
  if (cameraState.scenesByCamera[cameraId]) {
    return cameraState.scenesByCamera[cameraId];
  }
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/scenes`);
  const data = await res.json();
  if (!res.ok) throw new Error(apiErrorMessage(data, res.statusText));
  cameraState.scenesByCamera[cameraId] = data.scenes || [];
  return cameraState.scenesByCamera[cameraId];
}

function renderCollageTile(frame) {
  const figure = document.createElement("figure");
  figure.className = "collage-tile";
  const img = document.createElement("img");
  img.src = frame.image_url;
  img.alt = `${frame.slot} frame ${frame.frame_number}`;
  img.loading = "lazy";
  const caption = document.createElement("figcaption");
  caption.innerHTML = `<span class="slot-badge ${frame.slot}">${escapeHtml(frame.slot)}</span> frame ${frame.frame_number}`;
  figure.appendChild(img);
  figure.appendChild(caption);
  return figure;
}

function renderSceneRow(scene) {
  const row = document.createElement("article");
  row.className = "scene-row";

  const header = document.createElement("header");
  header.className = "scene-row-header";
  const range = formatSceneRange(scene.start_sec, scene.end_sec);
  const meta = [
    `Scene ${scene.scene_id}`,
    range,
    scene.cluster_id != null ? `cluster ${scene.cluster_id}` : "",
    scene.winner_share != null ? `${Math.round(scene.winner_share * 100)}% votes` : "",
    scene.pred_noise ? "noise" : "",
  ]
    .filter(Boolean)
    .join(" · ");
  header.textContent = meta;
  row.appendChild(header);

  const grid = document.createElement("div");
  grid.className = "scene-row-grid";
  (scene.frames || []).forEach((frame) => {
    grid.appendChild(renderCollageTile(frame));
  });
  row.appendChild(grid);
  return row;
}

async function renderCameraCollage(cameraIds) {
  const collage = cameraEl("camera-collage");
  const sceneCount = cameraEl("camera-scene-count");
  if (!collage) return;
  collage.innerHTML = "";
  if (!cameraIds.length) {
    if (sceneCount) sceneCount.textContent = "";
    return;
  }

  if (sceneCount) {
    sceneCount.textContent = `Loading ${cameraIds.length} camera(s)…`;
  }

  try {
    let totalScenes = 0;
    for (const cameraId of cameraIds) {
      const section = document.createElement("section");
      section.className = "camera-collage-section";
      const heading = document.createElement("h3");
      heading.textContent = cameraId;
      section.appendChild(heading);

      const scenes = await ensureScenesForCamera(cameraId);
      totalScenes += scenes.length;
      if (!scenes.length) {
        const empty = document.createElement("p");
        empty.className = "muted";
        empty.textContent = "No scenes for this camera.";
        section.appendChild(empty);
      } else {
        scenes.forEach((scene) => section.appendChild(renderSceneRow(scene)));
      }
      collage.appendChild(section);
    }
    if (sceneCount) {
      sceneCount.textContent = `${totalScenes} scene(s) across ${cameraIds.length} camera(s)`;
    }
  } catch (err) {
    if (sceneCount) sceneCount.textContent = "";
    showCameraError(err.message);
  }
}

function setHidden(id, hidden) {
  const el = cameraEl(id);
  if (!el) return;
  el.classList.toggle("hidden", hidden);
  if (hidden) {
    el.setAttribute("hidden", "");
  } else {
    el.removeAttribute("hidden");
  }
}

function renderScenePickers(cameraIds) {
  const scenePickers = cameraEl("camera-scene-pickers");
  if (!scenePickers) return;
  if (cameraIds.length < 2) {
    setHidden("camera-scene-pickers", true);
    setHidden("camera-compare-controls", true);
    return;
  }

  setHidden("camera-scene-pickers", false);
  setHidden("camera-compare-controls", false);
  scenePickers.innerHTML = "";

  cameraIds.forEach((cameraId) => {
    const row = document.createElement("div");
    row.className = "camera-scene-picker-row";
    const label = document.createElement("label");
    label.textContent = cameraId;
    const select = document.createElement("select");
    select.dataset.cameraId = cameraId;

    const scenes = cameraState.scenesByCamera[cameraId] || [];
    scenes.forEach((scene) => {
      const option = document.createElement("option");
      option.value = String(scene.scene_id);
      option.textContent = window.cameraCompare?.sceneLabel
        ? window.cameraCompare.sceneLabel(scene)
        : `Scene ${scene.scene_id}`;
      select.appendChild(option);
    });

    if (cameraState.selectedScenes[cameraId] != null) {
      select.value = String(cameraState.selectedScenes[cameraId]);
    } else if (scenes.length) {
      cameraState.selectedScenes[cameraId] = scenes[0].scene_id;
      select.value = String(scenes[0].scene_id);
    }

    select.addEventListener("change", () => {
      cameraState.selectedScenes[cameraId] = Number(select.value);
      updateCompareButton();
    });

    row.appendChild(label);
    row.appendChild(select);
    scenePickers.appendChild(row);
  });

  updateCompareButton();
}

function updateCompareButton() {
  const compareBtn = cameraEl("camera-compare-btn");
  if (!compareBtn) return;
  const selected = [...cameraState.selectedCameras];
  const ready = selected.length >= 2 && selected.every((id) => cameraState.selectedScenes[id] != null);
  compareBtn.disabled = !ready;
}

async function onCameraSelectionChanged() {
  const checkboxList = cameraEl("camera-checkbox-list");
  if (!checkboxList) return;

  const checked = [...checkboxList.querySelectorAll("input[type=checkbox]:checked")].map(
    (el) => el.value,
  );
  cameraState.selectedCameras = new Set(checked);

  for (const cameraId of checked) {
    await ensureScenesForCamera(cameraId);
    if (cameraState.selectedScenes[cameraId] == null) {
      const scenes = cameraState.scenesByCamera[cameraId] || [];
      if (scenes.length) cameraState.selectedScenes[cameraId] = scenes[0].scene_id;
    }
  }

  Object.keys(cameraState.selectedScenes).forEach((cameraId) => {
    if (!cameraState.selectedCameras.has(cameraId)) {
      delete cameraState.selectedScenes[cameraId];
    }
  });

  renderScenePickers(checked);
  await renderCameraCollage(checked);
  updateCompareButton();
}

async function loadCameras() {
  if (cameraState.loaded) return;
  try {
    const res = await fetch("/api/cameras");
    const data = await res.json();
    if (!res.ok) throw new Error(apiErrorMessage(data, res.statusText));

    cameraState.loaded = true;
    cameraState.cameraIds = data.camera_ids || [];
    cameraState.counts = data.counts || {};
    cameraState.clusterIds = data.camera_cluster_ids || {};
    cameraState.hasDebugArtifact = Boolean(data.has_debug_artifact);

    const debugBanner = cameraEl("camera-debug-banner");
    if (debugBanner) {
      if (!cameraState.hasDebugArtifact) {
        debugBanner.classList.remove("hidden");
        debugBanner.removeAttribute("hidden");
        debugBanner.textContent =
          "Distance comparison unavailable — re-run the cameras stage to generate camera_clustering_debug.npz.";
      } else {
        debugBanner.classList.add("hidden");
        debugBanner.setAttribute("hidden", "");
      }
    }

    renderCameraCheckboxes();

    if (cameraState.cameraIds.length === 1) {
      cameraState.selectedCameras.add(cameraState.cameraIds[0]);
      renderCameraCheckboxes();
      await onCameraSelectionChanged();
    }
  } catch (err) {
    showCameraError(err.message);
    throw err;
  }
}

function initCamerasTab() {
  if (camerasTabInitialized) return;
  if (!cameraEl("camera-checkbox-list")) return;
  camerasTabInitialized = true;

  cameraEl("camera-filter")?.addEventListener("input", () => renderCameraCheckboxes());
  cameraEl("camera-select-all")?.addEventListener("click", async () => {
    filteredCameraIds().forEach((id) => cameraState.selectedCameras.add(id));
    renderCameraCheckboxes();
    await onCameraSelectionChanged();
  });
  cameraEl("camera-clear-all")?.addEventListener("click", async () => {
    cameraState.selectedCameras.clear();
    renderCameraCheckboxes();
    await onCameraSelectionChanged();
    setHidden("camera-compare-panel", true);
  });
  cameraEl("camera-compare-btn")?.addEventListener("click", () => {
    const selections = [...cameraState.selectedCameras].map((cameraId) => ({
      camera_id: cameraId,
      scene_id: Number(cameraState.selectedScenes[cameraId]),
    }));
    const includeGlobal = Boolean(cameraEl("camera-include-global")?.checked);
    window.cameraCompare?.runComparison(selections, includeGlobal);
  });
}

window.initCamerasTab = initCamerasTab;
window.loadCameras = loadCameras;
