import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { compileExpr, fitChebyshev3D, PRESETS } from "./fit.js";
import { volumeVertex, volumeFragment } from "./shaders.js";

const MAX_N = 7;
const MAX_COEFFS = MAX_N * MAX_N * MAX_N;

const els = {
  preset: document.getElementById("preset"),
  expr: document.getElementById("expr"),
  deg: document.getElementById("deg"),
  scale: document.getElementById("scale"),
  steps: document.getElementById("steps"),
  tDeg: document.getElementById("tDeg"),
  half: document.getElementById("half"),
  resolve: document.getElementById("resolve"),
  mode: document.getElementById("mode"),
  profileStage: document.getElementById("profileStage"),
  fit: document.getElementById("fit"),
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
  if (els.tDeg) els.tDeg.value = "6";
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

const coeffData = new Float32Array(MAX_COEFFS);
const coeffTex = new THREE.DataTexture(
  coeffData,
  MAX_COEFFS,
  1,
  THREE.RedFormat,
  THREE.FloatType,
);
coeffTex.minFilter = THREE.NearestFilter;
coeffTex.magFilter = THREE.NearestFilter;
coeffTex.wrapS = THREE.ClampToEdgeWrapping;
coeffTex.wrapT = THREE.ClampToEdgeWrapping;
coeffTex.needsUpdate = true;

let worldMono = null;

const uniforms = {
  uCoeffTex: { value: coeffTex },
  uCoeffSize: { value: MAX_COEFFS },
  uDeg: { value: 4 },
  uHalf: { value: 2 },
  uScale: { value: 2.5 },
  uSteps: { value: 32 },
  uMode: { value: 0 },
  uTDeg: { value: 6 },
  uProfileStage: { value: 0 },
  uCameraPos: { value: new THREE.Vector3() },
  uAbsorbColor: { value: new THREE.Color(0.15, 0.25, 0.45) },
  uEmitColor: { value: new THREE.Color(0.55, 0.75, 1.0) },
};

const volumeMat = new THREE.ShaderMaterial({
  vertexShader: volumeVertex,
  fragmentShader: volumeFragment,
  uniforms,
  transparent: true,
  depthWrite: false,
  side: THREE.BackSide,
  // rgb is already integrated radiance (premultiplied); don't multiply by α again
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor,
  blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
});

const volumeMesh = new THREE.Mesh(new THREE.BoxGeometry(4, 4, 4), volumeMat);
scene.add(volumeMesh);

const boxHelper = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(4, 4, 4)),
  new THREE.LineBasicMaterial({ color: 0x3a4558 }),
);
scene.add(boxHelper);

function setBoxHalf(h) {
  const s = 2 * h;
  volumeMesh.geometry.dispose();
  volumeMesh.geometry = new THREE.BoxGeometry(s, s, s);
  boxHelper.geometry.dispose();
  boxHelper.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(s, s, s));
  uniforms.uHalf.value = h;
}

function fmtRel(v) {
  if (!Number.isFinite(v)) return "∞";
  if (v < 1e-3) return v.toExponential(2);
  return v.toPrecision(3);
}

function setErr(msg) {
  els.err.textContent = msg || "";
}

function modeLabel() {
  const stage = Number(els.profileStage?.value || 0);
  const base = els.mode.value === "pathc" ? "Path C Cheb-T" : "raymarch";
  return stage > 0 ? `${base} · P${stage}` : base;
}

let fps = 0;
let fpsFrames = 0;
let fpsLast = performance.now();
let cpuMsSmooth = 0;

function hudText() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const pr = renderer.getPixelRatio();
  const gpu = useGpuTimer ? `${gpuMsSmooth.toFixed(1)}ms gpu` : "gpu n/a";
  return `${modeLabel()} · ${Math.round(fps)} fps · ${cpuMsSmooth.toFixed(1)}ms cpu · ${gpu} · ${Math.round(w * pr)}×${Math.round(h * pr)}`;
}

function resize() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const res = Math.min(100, Math.max(40, Number(els.resolve?.value) || 85)) / 100;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(Math.round(w * res), Math.round(h * res), false);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  if (els.hud) els.hud.textContent = hudText();
}

/** World monomials once — no per-frame camera pullback. */
function uploadWorldCoeffs() {
  if (!worldMono) return;
  coeffData.fill(0);
  coeffData.set(worldMono);
  coeffTex.needsUpdate = true;
  if (els.basisMs) els.basisMs.textContent = "world · once";
}

function syncModeUniforms() {
  uniforms.uMode.value = els.mode.value === "pathc" ? 1 : 0;
  uniforms.uTDeg.value = Math.min(8, Math.max(2, Number(els.tDeg?.value) || 6));
  uniforms.uProfileStage.value = Math.min(4, Math.max(0, Number(els.profileStage?.value) || 0));
  els.modeLabel.textContent = modeLabel();
  resize();
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
    if (deg < 1 || deg > 6) throw new Error("poly deg must be 1…6");

    const fn = compileExpr(els.expr.value);
    const fit = fitChebyshev3D(fn, half, deg);

    worldMono = fit.mono;
    uploadWorldCoeffs();

    uniforms.uDeg.value = fit.deg;
    uniforms.uScale.value = densScale;
    uniforms.uSteps.value = steps;
    syncModeUniforms();
    setBoxHalf(half);

    const n = (fit.deg + 1) ** 3;
    els.fitErr.textContent = fmtRel(fit.fitRelL2);
    els.fitErr.className = "v " + (fit.fitRelL2 < 0.08 ? "ok" : "warn");
    els.nCoeff.textContent = String(n);
  } catch (e) {
    setErr(e instanceof Error ? e.message : String(e));
  }
}

els.preset.addEventListener("change", () => {
  applyPreset(els.preset.value);
  uploadFit();
});
els.fit.addEventListener("click", uploadFit);
els.mode.addEventListener("change", syncModeUniforms);
els.tDeg?.addEventListener("change", syncModeUniforms);
els.profileStage?.addEventListener("change", syncModeUniforms);
els.steps.addEventListener("change", () => {
  uniforms.uSteps.value = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
  resize();
});
els.resolve?.addEventListener("input", resize);
els.resolve?.addEventListener("change", resize);
els.scale.addEventListener("change", () => {
  uniforms.uScale.value = Number(els.scale.value) || 1;
});
els.reset.addEventListener("click", () => {
  camera.position.set(3.2, 2.4, 4.2);
  controls.target.set(0, 0, 0);
  controls.update();
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

function frame() {
  requestAnimationFrame(frame);
  const t0 = performance.now();
  fpsFrames++;
  if (t0 - fpsLast >= 500) {
    fps = (fpsFrames * 1000) / (t0 - fpsLast);
    fpsFrames = 0;
    fpsLast = t0;
    if (els.hud) els.hud.textContent = hudText();
    if (els.cpuMs) els.cpuMs.textContent = `${cpuMsSmooth.toFixed(2)} ms`;
  }

  controls.update();
  uniforms.uCameraPos.value.copy(camera.position);

  pollGpuTimer();
  if (useGpuTimer && !gpuQuery) {
    gpuQuery = gl.createQuery();
    gl.beginQuery(timerExt.TIME_ELAPSED_EXT, gpuQuery);
    gpuQueryActive = true;
  }

  renderer.render(scene, camera);

  if (gpuQueryActive && gpuQuery) {
    gl.endQuery(timerExt.TIME_ELAPSED_EXT);
    gpuQueryActive = false;
  }

  const dt = performance.now() - t0;
  cpuMsSmooth = cpuMsSmooth * 0.85 + dt * 0.15;
}
frame();
