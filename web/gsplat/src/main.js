import { loadSceneFromFile, sceneBounds, parseSceneJson } from "./loaders.js";

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d", { alpha: false });
const viewportEl = document.getElementById("viewport");
const statsEl = document.getElementById("stats");
const benchOut = document.getElementById("benchOut");
const hudEl = document.getElementById("hud");
const epsInput = document.getElementById("eps");
const epsVal = document.getElementById("epsVal");
const epsPreset = document.getElementById("epsPreset");
const resSelect = document.getElementById("res");
const liveToggle = document.getElementById("live");
const fileInput = document.getElementById("file");
const maxGaussians = document.getElementById("maxGaussians");
const blendModeSelect = document.getElementById("blendMode");
const maxInteractInput = document.getElementById("maxInteract");
const maxInteractVal = document.getElementById("maxInteractVal");
const maxInteractRow = document.getElementById("maxInteractRow");
const cdfModeSelect = document.getElementById("cdfMode");
const cdfModeRow = document.getElementById("cdfModeRow");

function formatMaxInteract(v) {
  return v <= 0 ? "off" : String(v);
}

function syncMaxInteractLabel() {
  maxInteractVal.textContent = formatMaxInteract(Number(maxInteractInput.value));
}

function currentMaxInteract() {
  return Number(maxInteractInput.value) | 0;
}

function currentCdfMode() {
  return cdfModeSelect?.value === "as" ? "as" : "logistic";
}

function formatCdfMode(m) {
  return m === "as" ? "A&S" : "logistic";
}

function syncApproxControls() {
  const approx = blendModeSelect.value === "approx";
  if (maxInteractRow) maxInteractRow.style.opacity = approx ? "1" : "0.45";
  if (maxInteractInput) maxInteractInput.disabled = !approx;
  if (cdfModeRow) cdfModeRow.style.opacity = approx ? "1" : "0.45";
  if (cdfModeSelect) cdfModeSelect.disabled = !approx;
}

/** Scale the bitmap to fill the viewport while keeping aspect ratio. */
function fitCanvasToViewport() {
  if (!viewportEl || !canvas.width || !canvas.height) return;
  const pad = 0;
  const vw = Math.max(1, viewportEl.clientWidth - pad);
  const vh = Math.max(1, viewportEl.clientHeight - pad);
  const scale = Math.min(vw / canvas.width, vh / canvas.height);
  canvas.style.width = `${Math.floor(canvas.width * scale)}px`;
  canvas.style.height = `${Math.floor(canvas.height * scale)}px`;
}

let yaw = 0.55;
let pitch = 0.22;
let radius = 4.2;
let target = [0, 0, 0];
let dragging = false;
let panning = false;
let lastX = 0;
let lastY = 0;
let dirty = true;

let reqId = 0;
let inflight = false;
let pending = false;
let lastFrameAt = performance.now();
let fpsEma = 0;
let sceneCache = null;
let sceneLabel = "demo";

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

function formatEps(eps) {
  if (eps <= 1e-12) return "off";
  if (eps >= 0.01) return eps.toPrecision(2);
  return eps.toExponential(0);
}

function syncEpsLabel() {
  epsVal.textContent = formatEps(Number(epsInput.value));
}

function currentEps() {
  if (epsPreset.value === "0") return 1e-12;
  return Number(epsInput.value);
}

function sizeFromSelect() {
  const [w, h] = resSelect.value.split("x").map(Number);
  return { width: w, height: h };
}

function orbitEye(y, p, r, tgt) {
  const cy = Math.cos(y);
  const sy = Math.sin(y);
  const cp = Math.cos(p);
  const sp = Math.sin(p);
  return [
    tgt[0] + r * cp * sy,
    tgt[1] + r * sp,
    tgt[2] + r * cp * cy,
  ];
}

function panTarget(dx, dy) {
  const eye = orbitEye(yaw, pitch, radius, target);
  const fx = target[0] - eye[0];
  const fy = target[1] - eye[1];
  const fz = target[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1;
  const f = [fx / fl, fy / fl, fz / fl];
  // right = forward × worldUp
  let rx = f[1] * 0 - f[2] * 1;
  let ry = f[2] * 0 - f[0] * 0;
  let rz = f[0] * 1 - f[1] * 0;
  let rl = Math.hypot(rx, ry, rz);
  if (rl < 1e-6) {
    // looking straight up/down — use world X as right
    rx = 1;
    ry = 0;
    rz = 0;
    rl = 1;
  }
  rx /= rl;
  ry /= rl;
  rz /= rl;
  // up = right × forward
  const ux = ry * f[2] - rz * f[1];
  const uy = rz * f[0] - rx * f[2];
  const uz = rx * f[1] - ry * f[0];
  const scale = radius * 0.0025;
  target[0] += (-rx * dx + ux * dy) * scale;
  target[1] += (-ry * dx + uy * dy) * scale;
  target[2] += (-rz * dx + uz * dy) * scale;
}

function applyCameraFit(scene) {
  const b = sceneBounds(scene);
  target = b.target;
  radius = b.radius;
  yaw = 0.55;
  pitch = 0.22;
}

function setScene(scene, label) {
  sceneCache = scene;
  sceneLabel = label;
  applyCameraFit(scene);
  dirty = true;
  statsEl.textContent = `Loading ${scene.count.toLocaleString()} Gaussians…`;
  worker.postMessage({ type: "scene", scene });
}

function requestRender(force = false) {
  if (!force && !dirty && !liveToggle.checked) return;
  if (inflight) {
    pending = true;
    return;
  }
  inflight = true;
  pending = false;
  dirty = false;
  const { width, height } = sizeFromSelect();
  worker.postMessage({
    type: "render",
    reqId: ++reqId,
    sentAt: performance.now(),
    camera: {
      eye: orbitEye(yaw, pitch, radius, target),
      target,
      up: [0, 1, 0],
      fovy: (45 * Math.PI) / 180,
      width,
      height,
    },
    opts: {
      eps: currentEps(),
      background: [0.03, 0.04, 0.06],
      blendMode: blendModeSelect.value,
      maxInteract: currentMaxInteract(),
      cdfMode: currentCdfMode(),
    },
  });
}

function showStats(stats, fps) {
  const extra =
    sceneCache?.totalInFile && sceneCache.totalInFile !== sceneCache.count
      ? `\nfile  ${sceneCache.totalInFile.toLocaleString()} → ${sceneCache.count.toLocaleString()}`
      : `\nsplats ${sceneCache?.count?.toLocaleString?.() ?? "?"}`;
  const lines = [
    `scene ${sceneLabel}`,
    `live  ${liveToggle.checked ? "on" : "off"}`,
    `blend ${stats.blendMode || blendModeSelect.value}`,
    `fps   ${fps.toFixed(1)}`,
    `ε     ${formatEps(stats.eps)}`,
    stats.blendMode === "approx" || blendModeSelect.value === "approx"
      ? `maxΦ  ${formatMaxInteract(stats.maxInteract ?? currentMaxInteract())}`
      : null,
    stats.blendMode === "approx" || blendModeSelect.value === "approx"
      ? `Φcdf  ${formatCdfMode(stats.cdfMode ?? currentCdfMode())}`
      : null,
    `frame ${stats.ms.toFixed(1)} ms`,
    `proj  ${stats.projected}`,
    `evals ${stats.splatEvals.toLocaleString()}`,
    `evals/px ${stats.evalsPerPixel.toFixed(2)}`,
    `early-out px ${stats.earlyOutPixels.toLocaleString()}`,
  ];
  statsEl.textContent = lines.filter(Boolean).join("\n") + extra;
  if (stats.radiusPx) {
    const r = stats.radiusPx;
    statsEl.textContent += `\nrPx   ${r.min.toFixed(1)}…${r.mean.toFixed(1)}…${r.max.toFixed(1)}`;
  }
  if (stats.profile) {
    const p = stats.profile;
    statsEl.textContent += [
      "",
      "── profile ──",
      `project  ${p.projectMs.toFixed(1)} ms`,
      `sort     ${p.sortMs.toFixed(1)} ms`,
      `tile     ${p.tileMs.toFixed(1)} ms`,
      `clear    ${p.clearMs.toFixed(1)} ms`,
      `raster   ${p.rasterMs.toFixed(1)} ms`,
      p.compositeMs
        ? `  ~gather    ${p.gatherMs.toFixed(1)} ms${p.phaseTimed ? " (meas.)" : ""}`
        : null,
      p.compositeMs
        ? `  ~composite ${p.compositeMs.toFixed(1)} ms${p.phaseTimed ? " (meas.)" : ""}`
        : null,
      `listTests ${p.listTests.toLocaleString()}`,
      stats.blendMode === "approx" && p.detail
        ? [
            `hits/px  ${p.hitMean.toFixed(1)} (max ${p.hitMax})`,
            `Φ-pairs  ${p.interactPairs.toLocaleString()}  beer ${p.beerPairs.toLocaleString()}  (max interact ${p.interactMax})`,
            `frontScans ${p.frontScans.toLocaleString()}  (pair tests; should drop with maxΦ)`,
            p.cappedPairs
              ? `capped  ${p.cappedPairs.toLocaleString()} overflow→Beer`
              : null,
            `T_slice  ${p.tSliceCalls.toLocaleString()} calls`,
            `T_iters  ${p.tSliceIters.toLocaleString()}  (pairwise overlap only)`,
            `cdfCalls ${p.cdfCalls.toLocaleString()}`,
            `appendix ${p.appendixCalls.toLocaleString()}`,
            `iters/px ${(p.pixelsComposited ? p.tSliceIters / p.pixelsComposited : 0).toFixed(1)}`,
          ]
            .filter(Boolean)
            .join("\n")
        : stats.blendMode === "approx" && p.hitMean
          ? `hits/px  ${p.hitMean.toFixed(1)} (max ${p.hitMax})  [Profile for overlap CDF counts]`
          : null,
    ]
      .filter(Boolean)
      .join("\n");
  }
  if (hudEl) {
    hudEl.textContent = `${(stats.blendMode || blendModeSelect.value)} · ${fps.toFixed(0)} fps · ${formatEps(stats.eps)} · ${stats.ms.toFixed(0)} ms`;
  }
}

worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "ready") {
    statsEl.textContent = `Loaded ${msg.count.toLocaleString()} Gaussians`;
    dirty = true;
    requestRender(true);
    return;
  }
  if (msg.type === "frame") {
    inflight = false;
    const rgba = new Uint8ClampedArray(msg.rgba);
    canvas.width = msg.width;
    canvas.height = msg.height;
    ctx.putImageData(new ImageData(rgba, msg.width, msg.height), 0, 0);
    fitCanvasToViewport();

    const now = performance.now();
    const dt = now - lastFrameAt;
    lastFrameAt = now;
    const fpsInst = dt > 0 ? 1000 / dt : 0;
    fpsEma = fpsEma ? fpsEma * 0.85 + fpsInst * 0.15 : fpsInst;
    showStats(msg.stats, fpsEma);

    if (pending || liveToggle.checked || dirty) requestRender(true);
  }
};

function maxCountOption() {
  return Number(maxGaussians.value) || 0;
}

async function loadDemoScene() {
  const res = await fetch("/scene.json");
  if (!res.ok) throw new Error("Missing public/scene.json — run npm run gen:scene");
  const scene = parseSceneJson(await res.json(), { maxCount: maxCountOption() });
  setScene(scene, "demo");
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  statsEl.textContent = `Parsing ${file.name}…`;
  try {
    const scene = await loadSceneFromFile(file, { maxCount: maxCountOption() });
    setScene(scene, file.name);
  } catch (err) {
    statsEl.textContent = `Upload failed: ${err.message || err}`;
  }
});

document.getElementById("resetScene").addEventListener("click", () => {
  fileInput.value = "";
  loadDemoScene().catch((err) => {
    statsEl.textContent = String(err.message || err);
  });
});

epsInput.addEventListener("input", () => {
  syncEpsLabel();
  const v = Number(epsInput.value);
  for (const opt of epsPreset.options) {
    if (Math.abs(Number(opt.value) - v) < 1e-9) {
      epsPreset.value = opt.value;
      break;
    }
  }
  dirty = true;
  requestRender(true);
});

epsPreset.addEventListener("change", () => {
  if (epsPreset.value === "0") {
    epsVal.textContent = "off";
  } else {
    epsInput.value = epsPreset.value;
    syncEpsLabel();
  }
  dirty = true;
  requestRender(true);
});

resSelect.addEventListener("change", () => {
  dirty = true;
  requestRender(true);
});

blendModeSelect.addEventListener("change", () => {
  syncApproxControls();
  dirty = true;
  requestRender(true);
});

maxInteractInput.addEventListener("input", () => {
  syncMaxInteractLabel();
  dirty = true;
  requestRender(true);
});

cdfModeSelect.addEventListener("change", () => {
  dirty = true;
  requestRender(true);
});

liveToggle.addEventListener("change", () => {
  if (liveToggle.checked) requestRender(true);
});

canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  // Right / middle / shift+left → pan (OrbitControls-style)
  panning = e.button === 2 || e.button === 1 || e.shiftKey;
  lastX = e.clientX;
  lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointerup", () => {
  dragging = false;
  panning = false;
  canvas.classList.remove("dragging");
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  if (panning) {
    panTarget(dx, dy);
  } else {
    yaw -= dx * 0.01;
    pitch = Math.max(-1.2, Math.min(1.2, pitch + dy * 0.01));
  }
  dirty = true;
  requestRender(true);
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const factor = Math.exp(e.deltaY * 0.0015);
    radius = Math.max(radius * 0.05, Math.min(radius * 40, radius * factor));
    dirty = true;
    requestRender(true);
  },
  { passive: false },
);

document.getElementById("bench").addEventListener("click", async () => {
  benchOut.textContent = "Running ε sweep…";
  const wasLive = liveToggle.checked;
  liveToggle.checked = false;
  await new Promise((r) => setTimeout(r, 80));

  if (!sceneCache) await loadDemoScene();
  const { renderFrame } = await import("./rasterizer.js");
  const { width, height } = sizeFromSelect();
  const camera = {
    eye: orbitEye(yaw, pitch, radius, target),
    target,
    up: [0, 1, 0],
    fovy: (45 * Math.PI) / 180,
    width,
    height,
  };

  const epsList = [1e-4, 1e-3, 1e-2, 5e-2];
  const lines = ["ε sweep (same view)", "ε        ms     evals    earlyPx   evals/px"];
  let baseline = null;
  let at1e2 = null;
  for (const eps of epsList) {
    renderFrame(sceneCache, camera, { eps, blendMode: blendModeSelect.value });
    const a = renderFrame(sceneCache, camera, { eps, blendMode: blendModeSelect.value });
    const b = renderFrame(sceneCache, camera, { eps, blendMode: blendModeSelect.value });
    const ms = (a.stats.ms + b.stats.ms) / 2;
    const s = a.stats;
    if (baseline === null) baseline = s.splatEvals;
    if (eps === 1e-2) at1e2 = s.splatEvals;
    lines.push(
      `${eps.toExponential(0).padEnd(8)} ${ms.toFixed(1).padStart(6)}  ${String(s.splatEvals).padStart(7)}  ${String(s.earlyOutPixels).padStart(8)}  ${s.evalsPerPixel.toFixed(2).padStart(7)}`,
    );
  }
  if (baseline && at1e2 != null) {
    lines.push(
      `evals drop vs 1e-4: ${(((baseline - at1e2) / baseline) * 100).toFixed(1)}% at ε=1e-2`,
    );
  }
  benchOut.textContent = lines.join("\n");
  liveToggle.checked = wasLive;
  dirty = true;
  requestRender(true);
});

document.getElementById("profile").addEventListener("click", async () => {
  benchOut.textContent = "Profiling alpha vs approx…";
  const wasLive = liveToggle.checked;
  liveToggle.checked = false;
  await new Promise((r) => setTimeout(r, 80));

  if (!sceneCache) await loadDemoScene();
  const { renderFrame } = await import("./rasterizer.js");
  const { width, height } = sizeFromSelect();
  const camera = {
    eye: orbitEye(yaw, pitch, radius, target),
    target,
    up: [0, 1, 0],
    fovy: (45 * Math.PI) / 180,
    width,
    height,
  };
  const eps = currentEps();
  const maxInteract = currentMaxInteract();
  const cdfMode = currentCdfMode();
  const optsBase = {
    eps,
    profile: true,
    background: [0.03, 0.04, 0.06],
    maxInteract,
    cdfMode,
  };

  function fmt(mode, s) {
    const p = s.profile;
    const lines = [
      `=== ${mode} ===`,
      `total   ${s.ms.toFixed(1)} ms  (raster ${p.rasterMs.toFixed(1)})`,
      `project ${p.projectMs.toFixed(1)}  sort ${p.sortMs.toFixed(1)}  tile ${p.tileMs.toFixed(1)}  clear ${p.clearMs.toFixed(1)}`,
      `listTests ${p.listTests.toLocaleString()}  evals ${s.splatEvals.toLocaleString()}  evals/px ${s.evalsPerPixel.toFixed(2)}`,
    ];
    if (mode === "approx") {
      lines.push(
        `hits/px ${p.hitMean.toFixed(2)} max ${p.hitMax}  maxΦ ${formatMaxInteract(maxInteract)}  Φcdf ${formatCdfMode(cdfMode)}`,
        `Φ-pairs ${p.interactPairs.toLocaleString()}  beer ${p.beerPairs.toLocaleString()}  capped ${p.cappedPairs.toLocaleString()}  maxInteract ${p.interactMax}`,
        `frontScans ${p.frontScans.toLocaleString()}  (pair tests)`,
        `T_slice calls ${p.tSliceCalls.toLocaleString()}  iters ${p.tSliceIters.toLocaleString()}`,
        `cdfCalls ${p.cdfCalls.toLocaleString()}  appendix ${p.appendixCalls.toLocaleString()}`,
        `iters/px ${(p.pixelsComposited ? p.tSliceIters / p.pixelsComposited : 0).toFixed(1)}  (pairwise overlap)`,
        `est. gather ${p.gatherMs.toFixed(1)} ms  composite ${p.compositeMs.toFixed(1)} ms${p.phaseTimed ? "  (measured)" : ""}`,
        `note: Appendix-A for nearest ≤maxΦ overlapping fronts; overflow+far → Beer; CDF=${formatCdfMode(cdfMode)}`,
      );
    }
    return lines.join("\n");
  }

  // warmup
  renderFrame(sceneCache, camera, { ...optsBase, blendMode: "alpha" });
  const alphaA = renderFrame(sceneCache, camera, { ...optsBase, blendMode: "alpha" });
  const alphaB = renderFrame(sceneCache, camera, { ...optsBase, blendMode: "alpha" });
  renderFrame(sceneCache, camera, { ...optsBase, blendMode: "approx" });
  const approxA = renderFrame(sceneCache, camera, { ...optsBase, blendMode: "approx" });
  const approxB = renderFrame(sceneCache, camera, { ...optsBase, blendMode: "approx" });

  const alpha = alphaA.stats.ms <= alphaB.stats.ms ? alphaA.stats : alphaB.stats;
  const approx = approxA.stats.ms <= approxB.stats.ms ? approxA.stats : approxB.stats;
  const slowdown = approx.ms / Math.max(alpha.ms, 1e-6);

  benchOut.textContent = [
    `Profile @ ${width}×${height}  ε=${formatEps(eps)}  maxΦ=${formatMaxInteract(maxInteract)}  Φcdf=${formatCdfMode(cdfMode)}  n=${sceneCache.count}`,
    fmt("alpha", alpha),
    "",
    fmt("approx", approx),
    "",
    `approx / alpha = ${slowdown.toFixed(2)}×`,
  ].join("\n");

  liveToggle.checked = wasLive;
  dirty = true;
  requestRender(true);
});

syncEpsLabel();
syncMaxInteractLabel();
syncApproxControls();
fitCanvasToViewport();
if (typeof ResizeObserver !== "undefined" && viewportEl) {
  new ResizeObserver(() => fitCanvasToViewport()).observe(viewportEl);
}
window.addEventListener("resize", fitCanvasToViewport);
loadDemoScene().catch((err) => {
  statsEl.textContent = String(err.message || err);
});
