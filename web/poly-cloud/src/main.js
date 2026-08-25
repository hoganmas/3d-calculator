import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { compileExpr, fitChebyshev3D, translateMonomial3D, PRESETS } from "./fit.js";
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
  mode: document.getElementById("mode"),
  fit: document.getElementById("fit"),
  reset: document.getElementById("reset"),
  err: document.getElementById("err"),
  fitErr: document.getElementById("fitErr"),
  nCoeff: document.getElementById("nCoeff"),
  fRange: document.getElementById("fRange"),
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
  els.steps.value = "48";
  if (els.tDeg) els.tDeg.value = "6";
}

applyPreset("blob");

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x07080b, 1);
els.viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
camera.position.set(3.2, 2.4, 4.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);
// Right-drag pans; prevent the browser context menu from eating it.
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
let fitDeg = 4;
const lastCam = new THREE.Vector3(NaN, NaN, NaN);

const uniforms = {
  uCoeffTex: { value: coeffTex },
  uCoeffSize: { value: MAX_COEFFS },
  uDeg: { value: 4 },
  uHalf: { value: 2 },
  uScale: { value: 2.5 },
  uSteps: { value: 48 },
  uMode: { value: 0 },
  uTDeg: { value: 6 },
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
  return els.mode.value === "pathc" ? "Path C Cheb-T" : "raymarch";
}

let fps = 0;
let fpsFrames = 0;
let fpsLast = performance.now();

function hudText() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const pr = renderer.getPixelRatio();
  return `${modeLabel()} · ${Math.round(fps)} fps · ${Math.round(w * pr)}×${Math.round(h * pr)} · dens deg ${uniforms.uDeg.value} · T deg ${uniforms.uTDeg.value}`;
}

function resize() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  if (els.hud) els.hud.textContent = hudText();
}

function uploadCameraBasis() {
  if (!worldMono) return;
  const o = camera.position;
  const camMono = translateMonomial3D(worldMono, fitDeg, o.x, o.y, o.z);
  coeffData.fill(0);
  coeffData.set(camMono);
  coeffTex.needsUpdate = true;
  lastCam.copy(o);
}

function syncModeUniforms() {
  uniforms.uMode.value = els.mode.value === "pathc" ? 1 : 0;
  uniforms.uTDeg.value = Math.min(8, Math.max(2, Number(els.tDeg?.value) || 6));
  els.modeLabel.textContent = modeLabel();
  resize();
}

function uploadFit() {
  setErr("");
  try {
    const half = Number(els.half.value);
    const deg = Number(els.deg.value);
    const densScale = Number(els.scale.value);
    const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 48));
    els.steps.value = String(steps);
    if (!(half > 0)) throw new Error("half-size must be > 0");
    if (deg < 1 || deg > 6) throw new Error("poly deg must be 1…6");

    const fn = compileExpr(els.expr.value);
    const fit = fitChebyshev3D(fn, half, deg);

    worldMono = fit.mono;
    fitDeg = fit.deg;
    lastCam.set(NaN, NaN, NaN);
    uploadCameraBasis();

    uniforms.uDeg.value = fit.deg;
    uniforms.uScale.value = densScale;
    uniforms.uSteps.value = steps;
    syncModeUniforms();
    setBoxHalf(half);

    const n = (fit.deg + 1) ** 3;
    els.fitErr.textContent = fmtRel(fit.fitRelL2);
    els.fitErr.className = "v " + (fit.fitRelL2 < 0.08 ? "ok" : "warn");
    els.nCoeff.textContent = String(n);
    els.fRange.textContent = `${fit.fMin.toPrecision(3)} / ${fit.fMax.toPrecision(3)}`;
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
els.steps.addEventListener("change", () => {
  uniforms.uSteps.value = Math.min(96, Math.max(8, Number(els.steps.value) || 48));
  resize();
});
els.scale.addEventListener("change", () => {
  uniforms.uScale.value = Number(els.scale.value) || 1;
});
els.reset.addEventListener("click", () => {
  camera.position.set(3.2, 2.4, 4.2);
  controls.target.set(0, 0, 0);
  controls.update();
  lastCam.set(NaN, NaN, NaN);
});

window.addEventListener("resize", resize);
resize();
uploadFit();

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  fpsFrames++;
  if (now - fpsLast >= 500) {
    fps = (fpsFrames * 1000) / (now - fpsLast);
    fpsFrames = 0;
    fpsLast = now;
    if (els.hud) els.hud.textContent = hudText();
  }
  controls.update();
  uniforms.uCameraPos.value.copy(camera.position);
  if (!lastCam.equals(camera.position)) {
    uploadCameraBasis();
  }
  renderer.render(scene, camera);
}
frame();
