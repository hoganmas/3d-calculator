import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { compileExpr, fitChebyshev3D, PRESETS } from "./fit.js";
import { volumeVertex, volumeFragment } from "./shaders.js";
import { clipGridVertex, clipGridFragment } from "./clipShaders.js";
import { bakeClipGridFibers, MAX_DEG } from "./clipGrid.js";

const MAX_N = 9;
const MAX_COEFFS = MAX_N * MAX_N * MAX_N;
const MAX_1D_N = 3 * MAX_DEG + 1;

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
let fitDeg = 4;
let clipDirty = true;
const lastBakeCam = new THREE.Vector3(NaN, NaN, NaN);
const lastBakeQuat = new THREE.Quaternion(0, 0, 0, 1);
let bakeMsSmooth = 0;

const uniforms = {
  uCoeffTex: { value: coeffTex },
  uCoeffSize: { value: MAX_COEFFS },
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
let lastBakeAt = 0;
const BAKE_MIN_MS = 120;

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
  const d = Math.min(8, Math.max(1, deg | 0));
  const next = {
    FIT_DEG: d,
    FIT_N: d + 1,
    FIT_1D: 3 * d,
    FIT_1D_N: 3 * d + 1,
  };
  const prev = volumeMat.defines || {};
  if (
    prev.FIT_DEG === next.FIT_DEG &&
    prev.FIT_N === next.FIT_N &&
    prev.FIT_1D === next.FIT_1D &&
    prev.FIT_1D_N === next.FIT_1D_N
  ) {
    return;
  }
  volumeMat.defines = { ...prev, ...next };
  volumeMat.needsUpdate = true;
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

function setBoxHalf(h) {
  const s = 2 * h;
  volumeMesh.geometry.dispose();
  volumeMesh.geometry = new THREE.BoxGeometry(s, s, s);
  boxHelper.geometry.dispose();
  boxHelper.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(s, s, s));
  uniforms.uHalf.value = h;
  clipUniforms.uHalf.value = h;
}

function fmtRel(v) {
  if (!Number.isFinite(v)) return "∞";
  if (v < 1e-3) return v.toExponential(2);
  return v.toPrecision(3);
}

function setErr(msg) {
  els.err.textContent = msg || "";
}

function isClipMode() {
  return els.mode.value === "clipgrid";
}

function modeLabel() {
  const stage = Number(els.profileStage?.value || 0);
  let base = "raymarch";
  if (els.mode.value === "pathc") base = "Path C Cheb-T";
  if (els.mode.value === "clipgrid") base = "clip-grid";
  return stage > 0 && !isClipMode() ? `${base} · P${stage}` : base;
}

let fps = 0;
let fpsFrames = 0;
let fpsLast = performance.now();
let cpuMsSmooth = 0;
/** Adaptive atlas edge while orbiting; grows/shrinks toward ~bake budget. */
let bakeMaxEdge = 112;
const BAKE_BUDGET_MS = 12;
const BAKE_EDGE_MIN = 64;
const BAKE_EDGE_MAX = 160;
const BAKE_EDGE_SETTLED = 192;
/** Display resolve scale on frames that also bake (keeps total frame time steadier). */
const BAKE_FRAME_DISPLAY_LOD = 0.5;
let restoreDisplayAfterBake = false;
let settleBake = false;

function hudText() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const pr = renderer.getPixelRatio();
  const gpu = useGpuTimer ? `${gpuMsSmooth.toFixed(1)}ms gpu` : "gpu n/a";
  const bake = isClipMode() ? ` · bake ${bakeMsSmooth.toFixed(0)}ms` : "";
  return `${modeLabel()} · ${Math.round(fps)} fps · ${cpuMsSmooth.toFixed(1)}ms cpu · ${gpu}${bake} · ${Math.round(w * pr)}×${Math.round(h * pr)}`;
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

function rebuildClipGrid({ settled = false } = {}) {
  if (!worldMono || !isClipMode()) return;
  const t0 = performance.now();
  const fbW = Math.max(1, renderer.domElement.width);
  const fbH = Math.max(1, renderer.domElement.height);
  const maxEdge = settled
    ? BAKE_EDGE_SETTLED
    : Math.min(bakeMaxEdge, BAKE_EDGE_MAX);
  const scale = Math.min(1, maxEdge / Math.max(fbW, fbH));
  const w = Math.max(1, Math.round(fbW * scale));
  const h = Math.max(1, Math.round(fbH * scale));
  camera.updateMatrixWorld(true);

  const baked = bakeClipGridFibers(
    worldMono,
    fitDeg,
    camera,
    w,
    h,
    clipUniforms.uHalf.value,
  );
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
  // FB size used for NDC mapping — keep full display intent via uniforms when
  // this frame is LOD'd; caller sets uFb* after restoring or to current buffer.
  clipUniforms.uFbW.value = fbW;
  clipUniforms.uFbH.value = fbH;
  clipUniforms.uNAlpha.value = baked.nAlpha;
  clipUniforms.uMax1d.value = baked.max1d;

  const M = baked.M;
  clipUniforms.uDirM.value.set(M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]);
  clipUniforms.uCameraPos.value.copy(camera.position);

  lastBakeCam.copy(camera.position);
  lastBakeQuat.copy(camera.quaternion);
  lastBakeAt = t0;
  clipDirty = false;
  const dt = performance.now() - t0;
  bakeMsSmooth = bakeMsSmooth * 0.7 + dt * 0.3;

  // Track interactive bake cost → next orbit bakes get smaller/larger atlases.
  if (!settled) {
    if (dt > BAKE_BUDGET_MS * 1.15) {
      bakeMaxEdge = Math.max(BAKE_EDGE_MIN, bakeMaxEdge - 16);
    } else if (dt < BAKE_BUDGET_MS * 0.7) {
      bakeMaxEdge = Math.min(BAKE_EDGE_MAX, bakeMaxEdge + 8);
    }
  }

  if (els.basisMs) {
    const tag = settled ? "hi" : `lod${bakeMaxEdge}`;
    els.basisMs.textContent = `bake ${bakeMsSmooth.toFixed(1)} ms · ${w}×${h} · ${tag}`;
  }
}

function syncModeUniforms() {
  const clip = isClipMode();
  volumeMesh.visible = !clip;
  clipQuad.visible = clip;
  uniforms.uMode.value = els.mode.value === "pathc" ? 1 : 0;
  uniforms.uTDeg.value = Math.min(8, Math.max(2, Number(els.tDeg?.value) || 6));
  uniforms.uProfileStage.value = Math.min(4, Math.max(0, Number(els.profileStage?.value) || 0));
  clipUniforms.uScale.value = uniforms.uScale.value;
  clipUniforms.uSteps.value = uniforms.uSteps.value;
  els.modeLabel.textContent = modeLabel();
  if (clip) {
    clipDirty = true;
    settleBake = true;
  }
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
    if (deg < 1 || deg > 8) throw new Error("poly deg must be 1…8");

    const fn = compileExpr(els.expr.value);
    const fit = fitChebyshev3D(fn, half, deg);

    worldMono = fit.mono;
    fitDeg = fit.deg;
    uploadWorldCoeffs();

    setShaderDegree(fit.deg);
    uniforms.uScale.value = densScale;
    uniforms.uSteps.value = steps;
    clipUniforms.uScale.value = densScale;
    clipUniforms.uSteps.value = steps;
    syncModeUniforms();
    setBoxHalf(half);
    clipDirty = true;

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
  const s = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
  uniforms.uSteps.value = s;
  clipUniforms.uSteps.value = s;
  resize();
});
els.resolve?.addEventListener("input", resize);
els.resolve?.addEventListener("change", resize);
els.scale.addEventListener("change", () => {
  const v = Number(els.scale.value) || 1;
  uniforms.uScale.value = v;
  clipUniforms.uScale.value = v;
});
els.reset.addEventListener("click", () => {
  camera.position.set(3.2, 2.4, 4.2);
  controls.target.set(0, 0, 0);
  controls.update();
  clipDirty = true;
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

function camMovedEnough() {
  if (clipDirty) return true;
  if (lastBakeCam.distanceToSquared(camera.position) > 1e-6) return true;
  if (Math.abs(lastBakeQuat.dot(camera.quaternion)) < 0.9995) return true;
  return false;
}

controls.addEventListener("end", () => {
  if (isClipMode()) {
    clipDirty = true;
    settleBake = true;
  }
});

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

  if (restoreDisplayAfterBake) {
    const { vw, vh, rw, rh } = displaySize(1);
    applyDisplaySize(rw, rh, vw, vh, { markClipDirty: false });
    clipUniforms.uFbW.value = rw;
    clipUniforms.uFbH.value = rh;
    restoreDisplayAfterBake = false;
  }

  // Rebake at most ~8 Hz while orbiting; hi-res settle on mouse-up / fit.
  if (isClipMode() && camMovedEnough()) {
    const due = clipDirty || t0 - lastBakeAt >= BAKE_MIN_MS;
    if (due) {
      const settled = settleBake;
      settleBake = false;
      try {
        // Bake frames: drop display resolve so bake+draw share the frame budget.
        if (!settled) {
          const { vw, vh, rw, rh } = displaySize(BAKE_FRAME_DISPLAY_LOD);
          applyDisplaySize(rw, rh, vw, vh, { markClipDirty: false });
          restoreDisplayAfterBake = true;
        }
        rebuildClipGrid({ settled });
        clipUniforms.uFbW.value = renderer.domElement.width;
        clipUniforms.uFbH.value = renderer.domElement.height;
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    }
  }

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
