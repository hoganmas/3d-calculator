import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { compileExpr, fitChebyshev3D, PRESETS } from "./fit.js";
import { volumeVertex, volumeFragment } from "./shaders.js";
import { clipGridVertex, clipGridFragment } from "./clipShaders.js";
import { bakeClipGridFibers, MAX_DEG } from "./clipGrid.js";
import {
  uploadResidentAtlas,
  bakeClipGridFibersGpu,
  initClipBakeGpu,
  isClipBakeGpuReady,
  isClipMarchReady,
  renderClipGridGpu,
  setClipGpuCanvasVisible,
  resizeClipGpuCanvas,
  ensurePipelinesForDegree,
  MAX_COEFFS,
} from "./clipBakeGpu.js";

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
const lastBakeCam = new THREE.Vector3(NaN, NaN, NaN);
const lastBakeQuat = new THREE.Quaternion(0, 0, 0, 1);
let bakeMsSmooth = 0;

const uniforms = {
  uCoeffTex: { value: coeffTex },
  uCoeffTexW: { value: COEFF_TEX_W },
  uCoeffTexH: { value: COEFF_TEX_H },
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
const BAKE_EDGE_MAX_CPU = 160;
const BAKE_EDGE_SETTLED_CPU = 192;
const BAKE_EDGE_MAX_GPU = 768;
const BAKE_EDGE_SETTLED_GPU = 768;
/** Soft cap on atlas bytes (CPU alloc + GPU upload). */ 
const ATLAS_BYTE_BUDGET = 48 * 1024 * 1024;
/** Display resolve scale on CPU bake frames (GPU bakes stay at full resolve). */
const BAKE_FRAME_DISPLAY_LOD = 0.5;
let restoreDisplayAfterBake = false;
let bakeInFlight = false;
let bakeBackend = "cpu";
/** Set while a sync settle is queued because a LOD bake is in flight. */
let pendingSettle = false;

function bakeEdgeLimits() {
  if (isClipBakeGpuReady()) {
    return { max: BAKE_EDGE_MAX_GPU, settled: BAKE_EDGE_SETTLED_GPU };
  }
  return { max: BAKE_EDGE_MAX_CPU, settled: BAKE_EDGE_SETTLED_CPU };
}

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
  return isClipMode() && isClipBakeGpuReady();
}

function syncClipPresentation() {
  const gpu = useGpuClipPath() && isClipMarchReady();
  clipQuad.visible = isClipMode() && !gpu;
  setClipGpuCanvasVisible(gpu);
  if (gpu) {
    const fbW = Math.max(1, renderer.domElement.width);
    const fbH = Math.max(1, renderer.domElement.height);
    if (resizeClipGpuCanvas(fbW, fbH)) {
      // Resized/cleared — redraw immediately so we never leave a blank overlay.
      const absorb = clipUniforms.uAbsorbColor.value;
      const emit = clipUniforms.uEmitColor.value;
      renderClipGridGpu({
        fbW,
        fbH,
        scale: clipUniforms.uScale.value,
        steps: clipUniforms.uSteps.value | 0,
        absorb: [absorb.r, absorb.g, absorb.b],
        emit: [emit.r, emit.g, emit.b],
      });
    }
  }
}

function presentBaked(baked, fbW, fbH, uploaded) {
  if (uploaded && baked.gpuResident) {
    setClipGpuCanvasVisible(true);
    clipQuad.visible = false;
    resizeClipGpuCanvas(fbW, fbH);
    const absorb = clipUniforms.uAbsorbColor.value;
    const emit = clipUniforms.uEmitColor.value;
    const ok = renderClipGridGpu({
      fbW,
      fbH,
      scale: clipUniforms.uScale.value,
      steps: clipUniforms.uSteps.value | 0,
      absorb: [absorb.r, absorb.g, absorb.b],
      emit: [emit.r, emit.g, emit.b],
    });
    if (!ok) {
      applyBakedAtlas(baked, fbW, fbH);
      setClipGpuCanvasVisible(false);
      clipQuad.visible = true;
    }
  } else {
    // CPU/WebGL path: never leave a blank WebGPU overlay on top.
    applyBakedAtlas(baked, fbW, fbH);
    setClipGpuCanvasVisible(false);
    clipQuad.visible = true;
  }
  clipUniforms.uFbW.value = fbW;
  clipUniforms.uFbH.value = fbH;
}

function clipAtlasSize(fbW, fbH, settled) {
  const lim = bakeEdgeLimits();
  let maxEdge = settled ? lim.settled : Math.min(bakeMaxEdge, lim.max);
  const nAlpha = 3 * Math.max(1, fitDeg) + 1;
  for (let guard = 0; guard < 12; guard++) {
    const scale = Math.min(1, maxEdge / Math.max(fbW, fbH));
    const w = Math.max(1, Math.round(fbW * scale));
    const h = Math.max(1, Math.round(fbH * scale));
    if (w * h * nAlpha * 4 <= ATLAS_BYTE_BUDGET) return { w, h, maxEdge };
    maxEdge = Math.max(BAKE_EDGE_MIN, Math.floor(maxEdge * 0.85));
  }
  const scale = Math.min(1, maxEdge / Math.max(fbW, fbH));
  return {
    w: Math.max(1, Math.round(fbW * scale)),
    h: Math.max(1, Math.round(fbH * scale)),
    maxEdge,
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

/**
 * Full bake + present. Prefers GPU tile-parallel Babbage; falls back to CPU f64.
 * Used for hi-res settle and fit — never fire-and-forget without bakeInFlight.
 */
async function rebuildClipGridSync({ settled = false } = {}) {
  if (!worldMono || !isClipMode()) return false;
  if (bakeInFlight) {
    if (settled) pendingSettle = true;
    return false;
  }
  bakeInFlight = true;
  const t0 = performance.now();
  try {
    const fbW = Math.max(1, renderer.domElement.width);
    const fbH = Math.max(1, renderer.domElement.height);
    const { w, h, maxEdge } = clipAtlasSize(fbW, fbH, settled);
    camera.updateMatrixWorld(true);

    let baked = null;
    let uploaded = false;
    bakeBackend = "cpu-babbage";

    if (isClipBakeGpuReady()) {
      try {
        const gpu = await bakeClipGridFibersGpu(
          worldMono,
          fitDeg,
          camera,
          w,
          h,
          clipUniforms.uHalf.value,
        );
        if (gpu) {
          baked = gpu;
          bakeBackend = "gpu-babbage";
          uploaded = true;
        }
      } catch (e) {
        console.warn("[clip-grid] GPU Babbage bake failed, CPU fallback", e);
      }
    }

    if (!baked) {
      baked = bakeClipGridFibers(
        worldMono,
        fitDeg,
        camera,
        w,
        h,
        clipUniforms.uHalf.value,
      );
      bakeBackend = "cpu-babbage";
      if (isClipBakeGpuReady()) {
        try {
          await ensurePipelinesForDegree(baked.deg);
          const up = uploadResidentAtlas(baked);
          if (up) {
            baked = up;
            bakeBackend = "babbage+gpu";
            uploaded = true;
          }
        } catch (e) {
          console.warn("[clip-grid] GPU upload failed, WebGL atlas fallback", e);
          bakeBackend = "cpu-babbage";
        }
      }
    }

    presentBaked(baked, fbW, fbH, uploaded);

    lastBakeCam.copy(camera.position);
    lastBakeQuat.copy(camera.quaternion);
    lastBakeAt = performance.now();
    clipDirty = false;
    const dt = performance.now() - t0;
    bakeMsSmooth = bakeMsSmooth * 0.7 + dt * 0.3;

    if (!settled && bakeBackend === "cpu-babbage") {
      if (dt > BAKE_BUDGET_MS * 1.15) {
        bakeMaxEdge = Math.max(BAKE_EDGE_MIN, bakeMaxEdge - 16);
      } else if (dt < BAKE_BUDGET_MS * 0.7) {
        bakeMaxEdge = Math.min(bakeEdgeLimits().max, bakeMaxEdge + 8);
      }
    } else if (bakeBackend === "gpu-babbage" || bakeBackend.startsWith("babbage")) {
      bakeMaxEdge = Math.min(
        bakeEdgeLimits().max,
        Math.max(bakeMaxEdge, 256),
      );
    }

    if (els.basisMs) {
      const tag = settled ? "hi" : `lod${Math.round(maxEdge)}`;
      const copy = uploaded ? "resident" : "tex";
      els.basisMs.textContent = `bake ${bakeMsSmooth.toFixed(1)} ms · ${w}×${h} · ${bakeBackend}/${copy} · ${tag}`;
    }
    return true;
  } catch (e) {
    console.error("[clip-grid] bake failed", e);
    setErr(e instanceof Error ? e.message : String(e));
    syncClipPresentation();
    return false;
  } finally {
    bakeInFlight = false;
    if (pendingSettle) {
      pendingSettle = false;
      void rebuildClipGridSync({ settled: true });
    }
  }
}

/** Hi-res settle: awaitable bake on this call stack. */
function runSettleBake() {
  if (!isClipMode() || !worldMono) return;
  cancelSettleTimer();
  void rebuildClipGridSync({ settled: true });
}

function syncModeUniforms() {
  const clip = isClipMode();
  volumeMesh.visible = !clip;
  uniforms.uMode.value = els.mode.value === "pathc" ? 1 : 0;
  uniforms.uTDeg.value = Math.min(8, Math.max(2, Number(els.tDeg?.value) || 6));
  uniforms.uProfileStage.value = Math.min(4, Math.max(0, Number(els.profileStage?.value) || 0));
  clipUniforms.uScale.value = uniforms.uScale.value;
  clipUniforms.uSteps.value = uniforms.uSteps.value;
  els.modeLabel.textContent = modeLabel();
  resize();
  if (clip) runSettleBake();
  else {
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

    // Mode sync + resize; settle once here (syncModeUniforms also settles on clip).
    const clip = isClipMode();
    volumeMesh.visible = !clip;
    uniforms.uMode.value = els.mode.value === "pathc" ? 1 : 0;
    uniforms.uTDeg.value = Math.min(8, Math.max(2, Number(els.tDeg?.value) || 6));
    uniforms.uProfileStage.value = Math.min(4, Math.max(0, Number(els.profileStage?.value) || 0));
    clipUniforms.uScale.value = densScale;
    clipUniforms.uSteps.value = steps;
    els.modeLabel.textContent = modeLabel();
    syncClipPresentation();
    resize();
    if (clip) runSettleBake();
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

/** Wheel fires end per notch; debounce so one sync settle runs after motion stops. */
const SETTLE_DEBOUNCE_MS = 180;
let settleTimer = 0;

function cancelSettleTimer() {
  if (settleTimer) {
    clearTimeout(settleTimer);
    settleTimer = 0;
  }
}

function scheduleSettleBake() {
  if (!isClipMode()) return;
  cancelSettleTimer();
  settleTimer = window.setTimeout(() => {
    settleTimer = 0;
    runSettleBake();
  }, SETTLE_DEBOUNCE_MS);
}

controls.addEventListener("start", () => {
  if (!isClipMode()) return;
  cancelSettleTimer();
  clipDirty = true;
});

controls.addEventListener("change", () => {
  if (!isClipMode()) return;
  clipDirty = true;
});

controls.addEventListener("end", () => {
  if (!isClipMode()) return;
  clipDirty = true;
  scheduleSettleBake();
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

  // LOD rebakes only while moving — hi-res settle is sync via runSettleBake().
  if (isClipMode() && !bakeInFlight && camMovedEnough()) {
    const due = clipDirty || t0 - lastBakeAt >= BAKE_MIN_MS;
    if (due) {
      const useGpu = isClipBakeGpuReady();
      if (!useGpu) {
        const { vw, vh, rw, rh } = displaySize(BAKE_FRAME_DISPLAY_LOD);
        applyDisplaySize(rw, rh, vw, vh, { markClipDirty: false });
        restoreDisplayAfterBake = true;
      }
      void rebuildClipGridSync({ settled: false });
    }
  }

  pollGpuTimer();
  if (useGpuTimer && !gpuQuery) {
    gpuQuery = gl.createQuery();
    gl.beginQuery(timerExt.TIME_ELAPSED_EXT, gpuQuery);
    gpuQueryActive = true;
  }

  renderer.render(scene, camera);

  if (useGpuClipPath() && isClipMarchReady()) {
    const absorb = clipUniforms.uAbsorbColor.value;
    const emit = clipUniforms.uEmitColor.value;
    renderClipGridGpu({
      fbW: renderer.domElement.width,
      fbH: renderer.domElement.height,
      scale: clipUniforms.uScale.value,
      steps: clipUniforms.uSteps.value | 0,
      absorb: [absorb.r, absorb.g, absorb.b],
      emit: [emit.r, emit.g, emit.b],
    });
  }

  if (gpuQueryActive && gpuQuery) {
    gl.endQuery(timerExt.TIME_ELAPSED_EXT);
    gpuQueryActive = false;
  }

  const dt = performance.now() - t0;
  cpuMsSmooth = cpuMsSmooth * 0.85 + dt * 0.15;
}
frame();

void initClipBakeGpu(els.viewport).then((ok) => {
  syncClipPresentation();
  if (ok && isClipMode()) {
    bakeMaxEdge = Math.max(bakeMaxEdge, 256);
    if (els.basisMs) els.basisMs.textContent = "gpu-babbage bake → march";
    runSettleBake();
  } else if (!ok && els.basisMs && isClipMode()) {
    els.basisMs.textContent = "cpu bake (no webgpu)";
    runSettleBake();
  }
});
