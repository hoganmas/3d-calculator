import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { compileExpr, fitChebyshev3D, PRESETS } from "./fit.js";
import { volumeVertex, volumeFragment } from "./shaders.js";
import { clipGridVertex, clipGridFragment } from "./clipShaders.js";
import { bakeClipGridFibers, MAX_DEG } from "./clipGrid.js";
import {
  initClipBakeGpu,
  isClipBakeGpuReady,
  isClipMarchReady,
  renderClipFrameGpu,
  setClipGpuCanvasVisible,
  resizeClipGpuCanvas,
  ensurePipelinesForDegree,
  getClipGpuProfile,
  MAX_COEFFS,
} from "./clipBakeGpu.js";

const els = {
  preset: document.getElementById("preset"),
  expr: document.getElementById("expr"),
  deg: document.getElementById("deg"),
  scale: document.getElementById("scale"),
  steps: document.getElementById("steps"),
  half: document.getElementById("half"),
  resolve: document.getElementById("resolve"),
  babbageTile: document.getElementById("babbageTile"),
  densFill: document.getElementById("densFill"),
  mode: document.getElementById("mode"),
  reset: document.getElementById("reset"),
  err: document.getElementById("err"),
  fitErr: document.getElementById("fitErr"),
  nCoeff: document.getElementById("nCoeff"),
  cpuMs: document.getElementById("cpuMs"),
  gpuMs: document.getElementById("gpuMs"),
  basisMs: document.getElementById("basisMs"),
  modeLabel: document.getElementById("modeLabel"),
  viewport: document.getElementById("viewport"),
  hud: document.getElementById("hud"),
  metricsDump: document.getElementById("metricsDump"),
  copyMetrics: document.getElementById("copyMetrics"),
};

for (const [key, p] of Object.entries(PRESETS)) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = p.label;
  els.preset.appendChild(opt);
}

function applyPreset(key) {
  const p = PRESETS[key] ?? PRESETS.blob;
  els.preset.value = key;
  els.expr.value = p.expr;
  els.deg.value = String(p.deg);
  els.scale.value = String(p.scale);
  els.half.value = String(p.half);
  els.steps.value = "32";
  if (els.resolve) els.resolve.value = "85";
}

applyPreset("blob");

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(1);
renderer.setClearColor(0x07080b, 1);
els.viewport.appendChild(renderer.domElement);

const gl = renderer.getContext();
const timerExt =
  gl.getExtension("EXT_disjoint_timer_query_webgl2") ||
  gl.getExtension("EXT_disjoint_timer_query");
const useGpuTimer = Boolean(timerExt && gl.createQuery);
let gpuQuery = null;
let gpuQueryActive = false;
let gpuMsSmooth = 0;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
camera.position.set(3.2, 2.4, 4.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

const COEFF_TEX_W = 64;
const COEFF_TEX_H = Math.ceil(MAX_COEFFS / COEFF_TEX_W);
const coeffData = new Float32Array(COEFF_TEX_W * COEFF_TEX_H);
const coeffTex = new THREE.DataTexture(
  coeffData,
  COEFF_TEX_W,
  COEFF_TEX_H,
  THREE.RedFormat,
  THREE.FloatType,
);
coeffTex.minFilter = THREE.NearestFilter;
coeffTex.magFilter = THREE.NearestFilter;
coeffTex.wrapS = THREE.ClampToEdgeWrapping;
coeffTex.wrapT = THREE.ClampToEdgeWrapping;
coeffTex.needsUpdate = true;

let worldMono = null;
let fitDeg = 4;
let clipDirty = true;
let bakeMsSmooth = 0;
let lastDensSubmitMs = 0;
let densSubmittedThisFrame = false;
let frameDtSmooth = 16;
let lastRafAt = 0;
/** Soft cap on atlas bytes. */
const ATLAS_BYTE_BUDGET = 48 * 1024 * 1024;
const BAKE_EDGE_MIN = 64;
/** Per-frame dens rebuild is expensive — keep modest atlas edge while orbiting. */
const BAKE_EDGE_MOVE = 256;
const BAKE_EDGE_SETTLE = 512;
/** Target JS→GPU submit cadence while orbiting. */
const FRAME_BUDGET_MS = 14;
let atlasEdge = BAKE_EDGE_MOVE;
let settleHiRes = false;
let settleTimer = 0;
/** CPU fallback only: min ms between rebakes while orbiting. */
const CPU_BAKE_MIN_MS = 120;
let lastCpuBakeAt = 0;
let cpuBakeInFlight = false;
let lastDensAtlasW = 0;
let lastDensAtlasH = 0;
let lastDensTile = 0;
let lastMetricsText = "";
let copyMetricsResetTimer = 0;

/** @returns {"auto" | "exact" | number} */
function readTileOverride() {
  const v = els.babbageTile?.value ?? "auto";
  if (v === "auto") return null;
  if (v === "exact") return "exact";
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** @returns {"chebyshev" | "newton"} */
function readDensFillMode() {
  return els.densFill?.value === "newton" ? "newton" : "chebyshev";
}

const uniforms = {
  uCoeffTex: { value: coeffTex },
  uCoeffTexW: { value: COEFF_TEX_W },
  uCoeffTexH: { value: COEFF_TEX_H },
  uHalf: { value: 2 },
  uScale: { value: 2.5 },
  uSteps: { value: 32 },
  uCameraPos: { value: new THREE.Vector3() },
  uAbsorbColor: { value: new THREE.Color(0.15, 0.25, 0.45) },
  uEmitColor: { value: new THREE.Color(0.55, 0.75, 1.0) },
};

const volumeMat = new THREE.ShaderMaterial({
  vertexShader: volumeVertex,
  fragmentShader: volumeFragment,
  uniforms,
  defines: {
    FIT_DEG: 4,
    FIT_N: 5,
    FIT_1D: 12,
    FIT_1D_N: 13,
  },
  transparent: true,
  depthWrite: false,
  side: THREE.BackSide,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor,
  blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
});

const alphaPlaceholder = new Float32Array(4);
const alphaTex = new THREE.DataTexture(alphaPlaceholder, 1, 1, THREE.RedFormat, THREE.FloatType);
alphaTex.minFilter = THREE.NearestFilter;
alphaTex.magFilter = THREE.NearestFilter;
alphaTex.generateMipmaps = false;
alphaTex.flipY = false;
alphaTex.colorSpace = THREE.NoColorSpace;
alphaTex.needsUpdate = true;

/** @type {THREE.DataTexture | null} */
let clipAtlasTex = null;
let clipAtlasW = 0;
let clipAtlasH = 0;

const clipUniforms = {
  uAlphaTex: { value: alphaTex },
  uGridW: { value: 1 },
  uGridH: { value: 1 },
  uFbW: { value: 1 },
  uFbH: { value: 1 },
  uNAlpha: { value: 1 },
  uMax1d: { value: 0 },
  uHalf: { value: 2 },
  uScale: { value: 2.5 },
  uTMid: { value: 0 },
  uTHw: { value: 1 },
  uSteps: { value: 32 },
  uCameraPos: { value: new THREE.Vector3() },
  uDirM: { value: new THREE.Matrix3() },
  uAbsorbColor: { value: new THREE.Color(0.15, 0.25, 0.45) },
  uEmitColor: { value: new THREE.Color(0.55, 0.75, 1.0) },
};

const clipMat = new THREE.ShaderMaterial({
  vertexShader: clipGridVertex,
  fragmentShader: clipGridFragment,
  uniforms: clipUniforms,
  defines: {
    CLIP_1D_N: 13,
  },
  transparent: true,
  depthWrite: false,
  depthTest: false,
  side: THREE.DoubleSide,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor,
  blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
});

function setShaderDegree(deg) {
  const d = Math.min(MAX_DEG, Math.max(1, deg | 0));
  const next = {
    FIT_DEG: d,
    FIT_N: d + 1,
    FIT_1D: 3 * d,
    FIT_1D_N: 3 * d + 1,
  };
  const prev = volumeMat.defines || {};
  if (
    prev.FIT_DEG !== next.FIT_DEG ||
    prev.FIT_N !== next.FIT_N ||
    prev.FIT_1D !== next.FIT_1D ||
    prev.FIT_1D_N !== next.FIT_1D_N
  ) {
    volumeMat.defines = { ...prev, ...next };
    volumeMat.needsUpdate = true;
  }
  const clipNext = { CLIP_1D_N: next.FIT_1D_N };
  const clipPrev = clipMat.defines || {};
  if (clipPrev.CLIP_1D_N !== clipNext.CLIP_1D_N) {
    clipMat.defines = { ...clipPrev, ...clipNext };
    clipMat.needsUpdate = true;
  }
}

const volumeMesh = new THREE.Mesh(new THREE.BoxGeometry(4, 4, 4), volumeMat);
scene.add(volumeMesh);

const clipQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), clipMat);
clipQuad.frustumCulled = false;
clipQuad.visible = false;
scene.add(clipQuad);

const boxHelper = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(4, 4, 4)),
  new THREE.LineBasicMaterial({ color: 0x3a4558 }),
);
scene.add(boxHelper);

/** World reference frame: unit grids + RGB axes (calculator-style). */
const worldGrid = new THREE.Group();
worldGrid.renderOrder = -1;
scene.add(worldGrid);

function makeAxisLabel(text, color, position) {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, size, size);
  ctx.font = "600 42px 'IBM Plex Sans', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(text, size / 2, size / 2 + 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: true,
    depthWrite: false,
  });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.35, 0.35, 0.35);
  spr.position.copy(position);
  return spr;
}

function styleGrid(grid, opacity) {
  const mats = Array.isArray(grid.material) ? grid.material : [grid.material];
  for (const m of mats) {
    m.transparent = true;
    m.opacity = opacity;
    m.depthWrite = false;
  }
}

function rebuildWorldGrid(half) {
  while (worldGrid.children.length) {
    const child = worldGrid.children.pop();
    child.geometry?.dispose?.();
    if (child.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const m of mats) {
        m.map?.dispose?.();
        m.dispose?.();
      }
    }
  }

  const h = Math.max(0.5, half);
  // Grid a bit past the fit box; aim for ~1 world-unit cells.
  const extent = Math.ceil(h + 0.5);
  const size = extent * 2;
  const divisions = Math.max(2, size);
  const major = 0x4a5568;
  const minor = 0x2a3140;

  const gridXZ = new THREE.GridHelper(size, divisions, major, minor);
  styleGrid(gridXZ, 0.55);
  worldGrid.add(gridXZ);

  const gridXY = new THREE.GridHelper(size, divisions, major, minor);
  gridXY.rotation.x = Math.PI / 2;
  styleGrid(gridXY, 0.35);
  worldGrid.add(gridXY);

  const gridYZ = new THREE.GridHelper(size, divisions, major, minor);
  gridYZ.rotation.z = Math.PI / 2;
  styleGrid(gridYZ, 0.35);
  worldGrid.add(gridYZ);

  const axisLen = extent + 0.25;
  const axisPositions = new Float32Array([
    0, 0, 0, axisLen, 0, 0, // +X
    0, 0, 0, 0, axisLen, 0, // +Y
    0, 0, 0, 0, 0, axisLen, // +Z
  ]);
  const axisColors = new Float32Array([
    0.9, 0.35, 0.38, 0.9, 0.35, 0.38,
    0.35, 0.75, 0.48, 0.35, 0.75, 0.48,
    0.4, 0.65, 0.95, 0.4, 0.65, 0.95,
  ]);
  const axisGeo = new THREE.BufferGeometry();
  axisGeo.setAttribute("position", new THREE.BufferAttribute(axisPositions, 3));
  axisGeo.setAttribute("color", new THREE.BufferAttribute(axisColors, 3));
  const axes = new THREE.LineSegments(
    axisGeo,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    }),
  );
  worldGrid.add(axes);

  const tip = extent + 0.45;
  worldGrid.add(makeAxisLabel("x", "#e85d66", new THREE.Vector3(tip, 0, 0)));
  worldGrid.add(makeAxisLabel("y", "#5ecf7a", new THREE.Vector3(0, tip, 0)));
  worldGrid.add(makeAxisLabel("z", "#6ea8fe", new THREE.Vector3(0, 0, tip)));
}

function setBoxHalf(h) {
  const s = 2 * h;
  volumeMesh.geometry.dispose();
  volumeMesh.geometry = new THREE.BoxGeometry(s, s, s);
  boxHelper.geometry.dispose();
  boxHelper.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(s, s, s));
  uniforms.uHalf.value = h;
  clipUniforms.uHalf.value = h;
  rebuildWorldGrid(h);
}

rebuildWorldGrid(2);

function fmtRel(v) {
  if (!Number.isFinite(v)) return "∞";
  if (v < 1e-3) return v.toExponential(2);
  return v.toPrecision(3);
}

function setErr(msg) {
  els.err.textContent = msg || "";
}

/** Highlight the expression field when compileExpr fails. */
function setExprCompileOk(ok) {
  if (!els.expr) return;
  els.expr.classList.toggle("invalid", !ok);
}

function syncExprCompileState() {
  try {
    compileExpr(els.expr.value);
    setExprCompileOk(true);
    setErr("");
    return true;
  } catch (e) {
    setExprCompileOk(false);
    const raw = (els.expr.value || "").trim();
    if (raw) setErr(e instanceof Error ? e.message : String(e));
    else setErr("");
    return false;
  }
}

function isClipMode() {
  return els.mode.value === "clipgrid";
}

function modeLabel() {
  if (els.mode.value === "clipgrid") return "clip-grid";
  return "raymarch";
}

let fps = 0;
let fpsFrames = 0;
let fpsLast = performance.now();
let cpuMsSmooth = 0;

function hudText() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const pr = renderer.getPixelRatio();
  // WebGL timer only sees Three (box/clear) — not WebGPU Babbage/march.
  const gl = useGpuTimer ? `${gpuMsSmooth.toFixed(1)}ms gl` : "gl n/a";
  let clip = "";
  if (isClipMode()) {
    const submit = densSubmittedThisFrame
      ? `submit ${bakeMsSmooth.toFixed(0)}ms`
      : `submit idle (last dens ${lastDensSubmitMs.toFixed(0)}ms)`;
    const p = getClipGpuProfile();
    const gpuSplit = p.timestamps
      ? ` · gpu seed ${p.seedMs.toFixed(1)}/fill ${p.fillMs.toFixed(1)}/march ${p.marchMs.toFixed(1)}`
      : "";
    clip = ` · rAF ${frameDtSmooth.toFixed(0)}ms · ${submit}${gpuSplit} · edge ${Math.round(atlasEdge)}`;
  }
  return `${modeLabel()} · ${Math.round(fps)} fps · ${cpuMsSmooth.toFixed(1)}ms js · ${gl}${clip} · ${Math.round(w * pr)}×${Math.round(h * pr)}`;
}

function buildMetricsReport() {
  const fbW = Math.max(1, renderer.domElement.width);
  const fbH = Math.max(1, renderer.domElement.height);
  const p = getClipGpuProfile();
  const lines = [
    `poly-cloud metrics  ${new Date().toISOString()}`,
    `mode            ${modeLabel()}`,
    `deg             ${fitDeg}`,
    `scale           ${uniforms.uScale.value}`,
    `steps           ${uniforms.uSteps.value}`,
    `half            ${uniforms.uHalf.value}`,
    `resolve%        ${els.resolve?.value ?? "—"}`,
    `framebuffer     ${fbW}×${fbH}`,
    `fps             ${Math.round(fps)}`,
    `rAF_ms          ${frameDtSmooth.toFixed(2)}`,
    `js_frame_ms     ${cpuMsSmooth.toFixed(2)}`,
    `webgl_ms        ${useGpuTimer ? gpuMsSmooth.toFixed(2) : "n/a"}`,
  ];
  if (isClipMode()) {
    lines.push(
      `dens_path       ${useGpuClipPath() ? "webgpu" : "cpu/webgl"}`,
      `dens_method     ${p.method || "—"}`,
      `dens_atlas      ${lastDensAtlasW}×${lastDensAtlasH}`,
      `dens_edge       ${Math.round(atlasEdge)}`,
      `dens_tile       ${lastDensTile || "—"}`,
      `dens_tiles_x    ${getClipGpuProfile().nTilesX || "—"}`,
      `dens_tile_ui    ${els.babbageTile?.value ?? "auto"}`,
      `dens_fill_ui    ${els.densFill?.value ?? "chebyshev"}`,
      `dens_submit_ms  ${densSubmittedThisFrame ? bakeMsSmooth.toFixed(2) : "idle"}`,
      `dens_last_ms    ${lastDensSubmitMs.toFixed(2)}`,
      `gpu_timestamps  ${p.timestamps ? "yes" : "no"}`,
      `gpu_seed_ms     ${p.timestamps ? p.seedMs.toFixed(3) : "n/a"}`,
      `gpu_fill_ms     ${p.timestamps ? p.fillMs.toFixed(3) : "n/a"}`,
      `gpu_march_ms    ${p.timestamps ? p.marchMs.toFixed(3) : "n/a"}`,
      `gpu_dens_ms     ${
        p.timestamps ? (p.seedMs + p.fillMs).toFixed(3) : "n/a"
      }`,
    );
  }
  lines.push(`fit_rel_L2      ${els.fitErr?.textContent ?? "—"}`);
  lines.push(`n_coeffs       ${els.nCoeff?.textContent ?? "—"}`);
  return lines.join("\n");
}

function refreshMetricsDump() {
  lastMetricsText = buildMetricsReport();
  if (els.metricsDump) els.metricsDump.value = lastMetricsText;
}

async function copyMetricsToClipboard() {
  const text = lastMetricsText || buildMetricsReport();
  try {
    await navigator.clipboard.writeText(text);
    if (els.copyMetrics) {
      els.copyMetrics.textContent = "Copied";
      if (copyMetricsResetTimer) clearTimeout(copyMetricsResetTimer);
      copyMetricsResetTimer = window.setTimeout(() => {
        if (els.copyMetrics) els.copyMetrics.textContent = "Copy";
      }, 1200);
    }
  } catch (e) {
    setErr(e instanceof Error ? e.message : "clipboard failed");
  }
}

function displaySize(lod = 1) {
  const vw = els.viewport.clientWidth;
  const vh = Math.max(els.viewport.clientHeight, 1);
  const res =
    (Math.min(100, Math.max(40, Number(els.resolve?.value) || 85)) / 100) * lod;
  return {
    vw,
    vh,
    rw: Math.max(1, Math.round(vw * res)),
    rh: Math.max(1, Math.round(vh * res)),
  };
}

function applyDisplaySize(rw, rh, vw, vh, { markClipDirty = true } = {}) {
  camera.aspect = vw / Math.max(vh, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(rw, rh, false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  if (markClipDirty) clipDirty = true;
  if (useGpuClipPath()) resizeClipGpuCanvas(rw, rh);
  if (els.hud) els.hud.textContent = hudText();
}

function resize() {
  const { vw, vh, rw, rh } = displaySize(1);
  applyDisplaySize(rw, rh, vw, vh, { markClipDirty: true });
}

function uploadWorldCoeffs() {
  if (!worldMono) return;
  coeffData.fill(0);
  coeffData.set(worldMono);
  coeffTex.needsUpdate = true;
  if (els.basisMs && !isClipMode()) els.basisMs.textContent = "world · once";
}

function configureAtlasTex(tex) {
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  // Critical: default flipY=true reverses atlas blocks → striped garbage.
  tex.flipY = false;
  tex.colorSpace = THREE.NoColorSpace;
}

function useGpuClipPath() {
  return isClipMode() && isClipBakeGpuReady() && isClipMarchReady();
}

function syncClipPresentation() {
  const gpu = useGpuClipPath();
  clipQuad.visible = isClipMode() && !gpu;
  setClipGpuCanvasVisible(gpu);
}

/** Atlas size from current edge target, shrunk to stay under the byte budget. */
function clipAtlasSize(fbW, fbH) {
  let maxEdge = Math.max(BAKE_EDGE_MIN, Math.round(atlasEdge));
  const nAlpha = 3 * Math.max(1, fitDeg) + 1;
  for (let guard = 0; guard < 12; guard++) {
    const scale = Math.min(1, maxEdge / Math.max(fbW, fbH));
    const w = Math.max(1, Math.round(fbW * scale));
    const h = Math.max(1, Math.round(fbH * scale));
    if (w * h * nAlpha * 4 <= ATLAS_BYTE_BUDGET) return { w, h };
    maxEdge = Math.max(BAKE_EDGE_MIN, Math.floor(maxEdge * 0.85));
  }
  const scale = Math.min(1, maxEdge / Math.max(fbW, fbH));
  return {
    w: Math.max(1, Math.round(fbW * scale)),
    h: Math.max(1, Math.round(fbH * scale)),
  };
}

function applyBakedAtlas(baked, fbW, fbH) {
  const texH = baked.height * baked.nAlpha;

  if (!clipAtlasTex || clipAtlasW !== baked.width || clipAtlasH !== texH) {
    if (clipAtlasTex) clipAtlasTex.dispose();
    clipAtlasTex = new THREE.DataTexture(
      baked.data,
      baked.width,
      texH,
      THREE.RedFormat,
      THREE.FloatType,
    );
    configureAtlasTex(clipAtlasTex);
    clipAtlasW = baked.width;
    clipAtlasH = texH;
    clipUniforms.uAlphaTex.value = clipAtlasTex;
  } else {
    clipAtlasTex.image.data.set(baked.data);
    clipAtlasTex.needsUpdate = true;
  }

  clipUniforms.uGridW.value = baked.width;
  clipUniforms.uGridH.value = baked.height;
  clipUniforms.uFbW.value = fbW;
  clipUniforms.uFbH.value = fbH;
  clipUniforms.uNAlpha.value = baked.nAlpha;
  clipUniforms.uMax1d.value = baked.max1d;
  clipUniforms.uTMid.value = baked.tMid;
  clipUniforms.uTHw.value = baked.tHw;

  const M = baked.M;
  clipUniforms.uDirM.value.set(M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]);
  clipUniforms.uCameraPos.value.copy(camera.position);
}

/** Per-frame GPU dens bake + march only when the view is dirty (scratch atlas). */
function drawClipGpuFrame() {
  densSubmittedThisFrame = false;
  if (!worldMono || !useGpuClipPath()) return false;
  // Camera still: keep last WebGPU canvas — submitting full dens every rAF is what
  // killed FPS while WebGL timers stayed ~0 (they never see WebGPU work).
  if (!clipDirty && !settleHiRes) return false;

  if (settleHiRes) {
    atlasEdge = BAKE_EDGE_SETTLE;
    settleHiRes = false;
  }

  const fbW = Math.max(1, renderer.domElement.width);
  const fbH = Math.max(1, renderer.domElement.height);
  const { w, h } = clipAtlasSize(fbW, fbH);
  camera.updateMatrixWorld(true);
  const absorb = clipUniforms.uAbsorbColor.value;
  const emit = clipUniforms.uEmitColor.value;
  const t0 = performance.now();
  const ok = renderClipFrameGpu({
    worldMono,
    deg: fitDeg,
    camera,
    width: w,
    height: h,
    half: clipUniforms.uHalf.value,
    fbW,
    fbH,
    scale: clipUniforms.uScale.value,
    steps: clipUniforms.uSteps.value | 0,
    absorb: [absorb.r, absorb.g, absorb.b],
    emit: [emit.r, emit.g, emit.b],
    tileOverride: readTileOverride(),
    fillMode: readDensFillMode(),
  });
  const submitMs = performance.now() - t0;
  if (ok) {
    densSubmittedThisFrame = true;
    lastDensSubmitMs = submitMs;
    lastDensAtlasW = w;
    lastDensAtlasH = h;
    const p = getClipGpuProfile();
    lastDensTile = p.tile || 0;
    bakeMsSmooth = bakeMsSmooth * 0.85 + submitMs * 0.15;
    clipDirty = false;
    // Adapt dens resolution to observed rAF spacing (GPU backlog shows up there).
    if (frameDtSmooth > FRAME_BUDGET_MS * 1.35) {
      atlasEdge = Math.max(BAKE_EDGE_MIN, atlasEdge * 0.85);
    } else if (frameDtSmooth < FRAME_BUDGET_MS * 0.75 && atlasEdge < BAKE_EDGE_MOVE) {
      atlasEdge = Math.min(BAKE_EDGE_MOVE, atlasEdge + 8);
    }
    if (els.basisMs) {
      const split = p.timestamps
        ? ` · seed ${p.seedMs.toFixed(1)}ms · fill ${p.fillMs.toFixed(1)}ms · march ${p.marchMs.toFixed(1)}ms`
        : " · gpu stamps n/a";
      els.basisMs.textContent = `gpu dens · ${w}×${h} · tile ${p.tile}×${p.nTilesX}${split} · ${p.method || ""}`;
    }
  }
  return ok;
}

/** CPU f64 Babbage → WebGL atlas (only when WebGPU clip path is unavailable). */
function rebuildClipCpuFallback() {
  if (!worldMono || !isClipMode() || useGpuClipPath() || cpuBakeInFlight) return;
  const tNow = performance.now();
  if (!clipDirty && tNow - lastCpuBakeAt < CPU_BAKE_MIN_MS) return;
  cpuBakeInFlight = true;
  const t0 = tNow;
  try {
    const fbW = Math.max(1, renderer.domElement.width);
    const fbH = Math.max(1, renderer.domElement.height);
    const { w, h } = clipAtlasSize(fbW, fbH);
    camera.updateMatrixWorld(true);
    const baked = bakeClipGridFibers(
      worldMono,
      fitDeg,
      camera,
      w,
      h,
      clipUniforms.uHalf.value,
    );
    applyBakedAtlas(baked, fbW, fbH);
    setClipGpuCanvasVisible(false);
    clipQuad.visible = true;
    clipDirty = false;
    lastCpuBakeAt = performance.now();
    lastDensAtlasW = w;
    lastDensAtlasH = h;
    bakeMsSmooth = bakeMsSmooth * 0.7 + (performance.now() - t0) * 0.3;
    if (els.basisMs) {
      els.basisMs.textContent = `cpu-babbage · ${w}×${h}`;
    }
  } catch (e) {
    console.error("[clip-grid] CPU bake failed", e);
    setErr(e instanceof Error ? e.message : String(e));
  } finally {
    cpuBakeInFlight = false;
  }
}

async function prepareClipGpuForDegree(deg) {
  if (!isClipBakeGpuReady()) return false;
  try {
    await ensurePipelinesForDegree(deg);
    syncClipPresentation();
    return true;
  } catch (e) {
    console.warn("[clip-grid] pipeline specialize failed", e);
    return false;
  }
}

function syncModeUniforms() {
  const clip = isClipMode();
  volumeMesh.visible = !clip;
  clipUniforms.uScale.value = uniforms.uScale.value;
  clipUniforms.uSteps.value = uniforms.uSteps.value;
  els.modeLabel.textContent = modeLabel();
  resize();
  syncClipPresentation();
  if (clip) {
    clipDirty = true;
    if (!useGpuClipPath()) rebuildClipCpuFallback();
  } else {
    setClipGpuCanvasVisible(false);
    clipQuad.visible = false;
  }
}

function uploadFit() {
  setErr("");
  try {
    const half = Number(els.half.value);
    const deg = Number(els.deg.value);
    const densScale = Number(els.scale.value);
    const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
    els.steps.value = String(steps);
    if (!(half > 0)) throw new Error("half-size must be > 0");
    if (deg < 1 || deg > MAX_DEG) throw new Error(`poly deg must be 1…${MAX_DEG}`);

    const fn = compileExpr(els.expr.value);
    setExprCompileOk(true);
    const fit = fitChebyshev3D(fn, half, deg);

    worldMono = fit.mono;
    fitDeg = fit.deg;
    uploadWorldCoeffs();

    setShaderDegree(fit.deg);
    uniforms.uScale.value = densScale;
    uniforms.uSteps.value = steps;
    clipUniforms.uScale.value = densScale;
    clipUniforms.uSteps.value = steps;
    setBoxHalf(half);

    const n = (fit.deg + 1) ** 3;
    els.fitErr.textContent = fmtRel(fit.fitRelL2);
    els.fitErr.className = "v " + (fit.fitRelL2 < 0.08 ? "ok" : "warn");
    els.nCoeff.textContent = String(n);

    // Mode sync + resize; GPU path draws every frame, CPU rebakes when dirty.
    const clip = isClipMode();
    volumeMesh.visible = !clip;
    clipUniforms.uScale.value = densScale;
    clipUniforms.uSteps.value = steps;
    els.modeLabel.textContent = modeLabel();
    resize();
    clipDirty = true;
    settleHiRes = true;
    void prepareClipGpuForDegree(fit.deg).then(() => {
      syncClipPresentation();
      if (clip && !useGpuClipPath()) {
        clipDirty = true;
        rebuildClipCpuFallback();
      }
    });
  } catch (e) {
    try {
      compileExpr(els.expr.value);
      setExprCompileOk(true);
    } catch {
      setExprCompileOk(false);
    }
    setErr(e instanceof Error ? e.message : String(e));
  }
}

/** Debounced Chebyshev refit when expr / deg / half become valid. */
let fitTimer = 0;
const FIT_DEBOUNCE_MS = 320;

function scheduleUploadFit(delay = FIT_DEBOUNCE_MS) {
  if (fitTimer) clearTimeout(fitTimer);
  fitTimer = window.setTimeout(() => {
    fitTimer = 0;
    if (!syncExprCompileState()) return;
    uploadFit();
  }, delay);
}

/** Scale / steps: no refit — update render uniforms only. */
function applyRenderHyperparams() {
  const densScale = Number(els.scale.value) || 1;
  const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
  els.steps.value = String(steps);
  uniforms.uScale.value = densScale;
  uniforms.uSteps.value = steps;
  clipUniforms.uScale.value = densScale;
  clipUniforms.uSteps.value = steps;
  clipDirty = true;
}

els.preset.addEventListener("change", () => {
  applyPreset(els.preset.value);
  if (fitTimer) clearTimeout(fitTimer);
  uploadFit();
});
els.expr.addEventListener("input", () => {
  syncExprCompileState();
  scheduleUploadFit();
});
els.deg.addEventListener("input", () => scheduleUploadFit(200));
els.deg.addEventListener("change", () => scheduleUploadFit(0));
els.half.addEventListener("input", () => scheduleUploadFit(200));
els.half.addEventListener("change", () => scheduleUploadFit(0));
els.scale.addEventListener("input", applyRenderHyperparams);
els.scale.addEventListener("change", applyRenderHyperparams);
els.mode.addEventListener("change", syncModeUniforms);
els.steps.addEventListener("input", applyRenderHyperparams);
els.steps.addEventListener("change", applyRenderHyperparams);
els.resolve?.addEventListener("input", resize);
els.resolve?.addEventListener("change", resize);
els.babbageTile?.addEventListener("change", () => {
  clipDirty = true;
  settleHiRes = true;
});
els.densFill?.addEventListener("change", () => {
  clipDirty = true;
  settleHiRes = true;
});
els.reset.addEventListener("click", () => {
  camera.position.set(3.2, 2.4, 4.2);
  controls.target.set(0, 0, 0);
  controls.update();
  clipDirty = true;
});
els.copyMetrics?.addEventListener("click", () => {
  void copyMetricsToClipboard();
});

window.addEventListener("resize", resize);
resize();
uploadFit();

function pollGpuTimer() {
  if (!useGpuTimer || !gpuQuery || gpuQueryActive) return;
  const available = gl.getQueryParameter(gpuQuery, gl.QUERY_RESULT_AVAILABLE);
  const disjoint = gl.getParameter(timerExt.GPU_DISJOINT_EXT);
  if (!available) return;
  if (!disjoint) {
    const ns = gl.getQueryParameter(gpuQuery, gl.QUERY_RESULT);
    const ms = ns / 1e6;
    gpuMsSmooth = gpuMsSmooth * 0.8 + ms * 0.2;
    if (els.gpuMs) els.gpuMs.textContent = `${gpuMsSmooth.toFixed(2)} ms`;
  }
  gl.deleteQuery(gpuQuery);
  gpuQuery = null;
}

controls.addEventListener("start", () => {
  if (!isClipMode()) return;
  if (settleTimer) {
    clearTimeout(settleTimer);
    settleTimer = 0;
  }
  atlasEdge = Math.min(atlasEdge, BAKE_EDGE_MOVE);
  clipDirty = true;
});

controls.addEventListener("change", () => {
  if (isClipMode()) clipDirty = true;
});

controls.addEventListener("end", () => {
  if (!isClipMode()) return;
  clipDirty = true;
  if (settleTimer) clearTimeout(settleTimer);
  // One higher-res dens rebuild after motion stops (not a resident cache).
  settleTimer = window.setTimeout(() => {
    settleTimer = 0;
    settleHiRes = true;
    clipDirty = true;
  }, 160);
});

function frame() {
  requestAnimationFrame(frame);
  const t0 = performance.now();
  if (lastRafAt > 0) {
    frameDtSmooth = frameDtSmooth * 0.85 + (t0 - lastRafAt) * 0.15;
  }
  lastRafAt = t0;
  fpsFrames++;
  if (t0 - fpsLast >= 500) {
    fps = (fpsFrames * 1000) / (t0 - fpsLast);
    fpsFrames = 0;
    fpsLast = t0;
    if (els.hud) els.hud.textContent = hudText();
    if (els.cpuMs) {
      els.cpuMs.textContent = `${cpuMsSmooth.toFixed(2)} ms js · ${frameDtSmooth.toFixed(1)} ms rAF`;
    }
    refreshMetricsDump();
  }

  controls.update();
  uniforms.uCameraPos.value.copy(camera.position);

  if (isClipMode() && worldMono && !useGpuClipPath()) {
    rebuildClipCpuFallback();
  }

  pollGpuTimer();
  if (useGpuTimer && !gpuQuery) {
    gpuQuery = gl.createQuery();
    gl.beginQuery(timerExt.TIME_ELAPSED_EXT, gpuQuery);
    gpuQueryActive = true;
  }

  renderer.render(scene, camera);

  if (isClipMode() && worldMono && useGpuClipPath()) {
    drawClipGpuFrame();
  }

  if (gpuQueryActive && gpuQuery) {
    gl.endQuery(timerExt.TIME_ELAPSED_EXT);
    gpuQueryActive = false;
  }

  const dt = performance.now() - t0;
  cpuMsSmooth = cpuMsSmooth * 0.85 + dt * 0.15;
}
frame();

void initClipBakeGpu(els.viewport).then(async (ok) => {
  if (ok && worldMono) await prepareClipGpuForDegree(fitDeg);
  syncClipPresentation();
  clipDirty = true;
  settleHiRes = true;
  if (els.basisMs && isClipMode()) {
    els.basisMs.textContent = ok
      ? "gpu dens (dirty frames)"
      : "cpu bake (no webgpu)";
  }
  if (isClipMode() && !useGpuClipPath()) {
    rebuildClipCpuFallback();
  }
});
