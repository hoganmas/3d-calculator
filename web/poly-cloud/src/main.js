import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "mathlive";
import "mathlive/static.css";
import "./theme.css";
import { initTheme, onThemeChange, readThemeColors, getThemePref, setThemePref } from "./theme.js";
import { createLavaBackground } from "./lavaBackground.js";
import { compileExpr, classifyExpr, fitChebyshev3D, PRESETS, formatParamLatexValue, compileParamLatex } from "./fit.js";
import {
  syncParamsFromDefinitions,
  applyParamSeed,
  getParamValues,
  listParamNames,
  getParam,
  recompileAllParams,
  evalParamEquations,
  tickParamAnimation,
  anyParamNeedsTick,
  collectAnimDirtyParams,
} from "./params.js";
import {
  listExpressions,
  setExpressions,
  updateExprSilent,
  hexToRgb01,
  resolveExprRole,
  setExpressionsOnChange,
  replaceExprWarnings,
  getExprWarning,
  insertExprAt,
  removeExprSilent,
  commitAutoParams,
  resolveExprGradient,
  color2ForPrimary,
} from "./expressions.js";
import { mountExprList } from "./exprListUi.js";
import { clipGridVertex, clipGridFragment } from "./clipShaders.js";
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix, MAX_DEG } from "./clipGrid.js";
import { idctCheb3D, idctChebGrad3D } from "./chebIdct.js";
import {
  beginKeyframePass,
  clearKeyframeCaches,
  getKeyframeMetrics,
  keyframeAnimParam,
  noteKeyframeLayer,
  ensureLayerKeyframes,
  sampleLayerKeyframes,
  peekKeyframeBlend,
} from "./animKeyframes.js";
import {
  initClipBakeGpu,
  isClipBakeGpuReady,
  isClipMarchReady,
  renderClipFrameGpu,
  setClipGpuCanvasVisible,
  ensurePipelinesForDegree,
  getClipGpuProfile,
  resetClipGpuProfile,
  uploadSceneVolumes,
  uploadSceneColors,
  hasUploadedVolume,
  setConstraintKeyframeBlends,
  resizeClipGpuCanvas,
  clearClipGpuFrame,
  syncClipGpuWorldGrid,
  applyClipGpuTheme,
} from "./clipBakeGpu.js";

initTheme();

const els = {
  preset: document.getElementById("preset"),
  exprList: document.getElementById("exprList"),
  deg: document.getElementById("deg"),
  scale: document.getElementById("scale"),
  steps: document.getElementById("steps"),
  boxSize: document.getElementById("boxSize"),
  marchDownscale: document.getElementById("marchDownscale"),
  marchScaleLabel: document.getElementById("marchScaleLabel"),
  reset: document.getElementById("reset"),
  err: document.getElementById("err"),
  viewport: document.getElementById("viewport"),
  hud: document.getElementById("hud"),
  metricsDump: document.getElementById("metricsDump"),
  copyMetrics: document.getElementById("copyMetrics"),
  openSettings: document.getElementById("openSettings"),
  closeSettings: document.getElementById("closeSettings"),
  settingsDialog: document.getElementById("settingsDialog"),
  themePref: document.getElementById("themePref"),
};

for (const [key, p] of Object.entries(PRESETS)) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = p.label;
  els.preset.appendChild(opt);
}

function applyPreset(key) {
  const p = PRESETS[key] ?? PRESETS.sincos;
  els.preset.value = key in PRESETS ? key : "sincos";
  pendingParamSeed = p.params ?? {};
  if (Array.isArray(p.expressions) && p.expressions.length) {
    setExpressions(p.expressions);
  } else {
    setExpressions([{ latex: p.latex }]);
  }
  exprListApi?.render();
}

/** Preset param defaults applied on next successful compile/sync. */
let pendingParamSeed = {};

/** @type {{ render: () => void, syncAllParamSliders?: () => void, syncParamChrome?: () => boolean } | null} */
let exprListApi = null;

function fmtParamNum(v) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1000 || a < 0.01)) return v.toPrecision(3);
  return String(Math.round(v * 1000) / 1000);
}

/**
 * Insert missing `name=value` expression rows at the bottom (sorted) without UI notify.
 * @param {string[]} names
 */
function ensureParamExprRows(names) {
  const missing = [...new Set(names)].filter(Boolean).sort();
  if (!missing.length) return false;
  for (const name of missing) {
    const seed = pendingParamSeed[name] ?? {};
    const value = Number.isFinite(seed.value) ? seed.value : 1;
    insertExprAt(listExpressions().length, {
      latex: `${name}=${formatParamLatexValue(value)}`,
      sliderMin: seed.min,
      sliderMax: seed.max,
      sliderSpeed: seed.speed,
      sliderAnimating: !!(seed.animate ?? seed.animating),
      sliderPhase: seed.phase,
      autoParam: true,
    });
  }
  return true;
}

/** Names referenced by field free-symbols or parameter RHS deps. */
function collectParamReferences() {
  /** @type {Set<string>} */
  const refs = new Set();
  for (const item of listExpressions()) {
    if (!item.enabled || !String(item.latex || "").trim()) continue;
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") {
        const compiled = compileParamLatex(item.latex, classified.paramName);
        for (const p of compiled.freeParams) refs.add(p);
      } else {
        const compiled = compileExpr(item.latex);
        for (const p of compiled.freeParams) refs.add(p);
      }
    } catch {
      /* ignore */
    }
  }
  return refs;
}

/**
 * Drop ephemeral auto-param rows that are no longer referenced (typing undo).
 * Committed rows (after blur) are kept.
 */
function pruneUnusedAutoParams() {
  const refs = collectParamReferences();
  let removed = false;
  for (const item of listExpressions()) {
    if (!item.autoParam) continue;
    let name = null;
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") name = classified.paramName;
    } catch {
      continue;
    }
    if (!name || refs.has(name)) continue;
    if (removeExprSilent(item.id)) removed = true;
  }
  return removed;
}

/**
 * Compile all expressions: parameter rows feed shared values; field rows become layers.
 * Free symbols without a dedicated `a=…` row get an auto-created parameter line.
 * @param {{ rebuildUi?: boolean, _afterEnsure?: boolean }} [opts]
 */
function compileAllExprs(opts = {}) {
  const rebuildUi = opts.rebuildUi !== false;
  const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());

  /** @type {{ item: any, name: string }[]} */
  const paramRows = [];
  /** @type {{ item: any, compiled: any, fn: Function, role: string }[]} */
  const layers = [];
  const freeSet = new Set();
  const definedParams = new Set();
  /** @type {[string, string][]} */
  const warnings = [];
  replaceExprWarnings([]);

  for (const item of items) {
    let classified;
    try {
      classified = classifyExpr(item.latex);
    } catch {
      continue;
    }
    if (classified.kind === "parameter") {
      const name = classified.paramName;
      if (!name) continue;
      if (definedParams.has(name)) {
        warnings.push([item.id, `Variable “${name}” is already declared`]);
        continue;
      }
      definedParams.add(name);
      paramRows.push({ item, name });
      continue;
    }
    const compiled = compileExpr(item.latex);
    for (const p of compiled.freeParams) freeSet.add(p);
    // Spatially constant (0th-order) fields: do not graph.
    if (!compiled.usesSpace || compiled.shade === "none") continue;
    const role = resolveExprRole(item.role, compiled.kind);
    layers.push({
      item,
      compiled,
      role,
      fn: compiled.bind(getParamValues()),
    });
  }

  replaceExprWarnings(warnings);

  /** @type {Parameters<typeof syncParamsFromDefinitions>[0]} */
  const defs = paramRows.map(({ item, name }) => ({
    name,
    latex: item.latex,
    exprId: item.id,
    hosted: false,
    min: item.sliderMin,
    max: item.sliderMax,
    speed: item.sliderSpeed,
    animating: item.sliderAnimating,
    phase: item.sliderPhase,
  }));

  syncParamsFromDefinitions(defs, pendingParamSeed);

  // Param-equation deps (a=b+1) without their own row.
  const depNames = recompileAllParams();
  const known = new Set(defs.map((d) => d.name));
  const needRows = [
    ...[...freeSet].filter((n) => !definedParams.has(n)),
    ...[...new Set(depNames)].filter((n) => !known.has(n) && !definedParams.has(n)),
  ];

  const pruned = !opts._afterEnsure && pruneUnusedAutoParams();
  if ((pruned || needRows.length) && !opts._afterEnsure) {
    if (needRows.length) ensureParamExprRows(needRows);
    return compileAllExprs({ ...opts, _afterEnsure: true });
  }

  if (Object.keys(pendingParamSeed).length) {
    applyParamSeed(pendingParamSeed);
    for (const { item, name } of paramRows) {
      const p = getParam(name);
      if (!p) continue;
      updateExprSilent(item.id, {
        latex: p.latex,
        sliderMin: p.min,
        sliderMax: p.max,
        sliderSpeed: p.speed,
        sliderAnimating: p.animating,
        sliderPhase: p.phase,
      });
    }
    pendingParamSeed = {};
  }

  evalParamEquations(performance.now() / 1000);
  const params = getParamValues();
  for (const L of layers) L.fn = L.compiled.bind(params);

  if (rebuildUi) {
    // Row inserts or kind flips: prefer full render when chrome can't sync in place.
    if (!exprListApi?.syncParamChrome?.()) {
      exprListApi?.render();
    }
  }

  const nCons = layers.filter((L) => L.role === "constraint").length;
  const nDens = layers.filter((L) => L.role === "density").length;
  lastExprMeta = {
    kind: nCons && nDens ? "mixed" : nCons ? "constraint" : "bare",
    shade: nCons && !nDens ? "iso" : "volume",
    isoLevel: 0,
    label: `${nDens} density · ${nCons} manifold`,
  };

  return {
    freeParams: [...freeSet].sort(),
    layers,
    warnings: warnings.map(([, msg]) => msg),
  };
}

/** Last successful classify/compile summary. */
let lastExprMeta = {
  kind: "bare",
  shade: "volume",
  isoLevel: 0,
  label: "expression → volume",
};

const MARCH_DOWNSCALE_MIN = 1;
const MARCH_DOWNSCALE_MAX = 16;
/** Label only these notches (every integer still snaps). */
const MARCH_DOWNSCALE_LABELS = new Set([1, 2, 4, 8, 16]);

function marchDownscaleTickPct(n) {
  return ((n - MARCH_DOWNSCALE_MIN) / (MARCH_DOWNSCALE_MAX - MARCH_DOWNSCALE_MIN)) * 100;
}

function initMarchSliderUi() {
  const ticks = document.getElementById("marchSliderTicks");
  const labels = document.getElementById("marchSliderLabels");
  if (!ticks || !labels) return;
  ticks.replaceChildren();
  labels.replaceChildren();
  for (let n = MARCH_DOWNSCALE_MIN; n <= MARCH_DOWNSCALE_MAX; n++) {
    const pct = `${marchDownscaleTickPct(n)}%`;
    const tick = document.createElement("span");
    tick.style.setProperty("--tick", pct);
    ticks.appendChild(tick);
    if (MARCH_DOWNSCALE_LABELS.has(n)) {
      const lab = document.createElement("span");
      lab.style.setProperty("--tick", pct);
      lab.textContent = `${n}×`;
      labels.appendChild(lab);
    }
  }
}

function readMarchDownscale() {
  if (!els.marchDownscale) return 1;
  const n = Math.round(Number(els.marchDownscale.value) || 1);
  return Math.min(MARCH_DOWNSCALE_MAX, Math.max(MARCH_DOWNSCALE_MIN, n));
}

function syncMarchSlider() {
  const n = readMarchDownscale();
  if (els.marchDownscale) els.marchDownscale.value = String(n);
  if (els.marchScaleLabel) els.marchScaleLabel.textContent = `${n}×`;
  return n;
}

function marchDownscale() {
  return readMarchDownscale();
}

applyPreset("sincos");
if (!listExpressions().length) {
  setExpressions([{ latex: PRESETS.sincos.latex }]);
}
initMarchSliderUi();
syncMarchSlider();

function layerRgbFromItem(item) {
  const grad = resolveExprGradient(item);
  const colors = grad.colors.map((hex) => hexToRgb01(hex));
  return {
    color: colors[0],
    color2: colors[colors.length - 1],
    colors,
  };
}

exprListApi = mountExprList({
  root: els.exprList,
  onExprChange: () => {
    syncExprCompileState();
    scheduleUploadFit();
  },
  onParamChange: () => {
    syncExprCompileState();
    scheduleUploadFit(80);
  },
  onColorChange: () => {
    // Colors only — skip Chebyshev refit; push RGB to GPU dens layers + constraints.
    if (lastSceneBake) {
      const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());
      const densCols = [];
      const consCols = [];
      for (const item of items) {
        let compiled;
        try {
          if (classifyExpr(item.latex).kind === "parameter") continue;
          compiled = compileExpr(item.latex);
        } catch {
          continue;
        }
        if (!compiled.usesSpace) continue;
        const role = resolveExprRole(item.role, compiled.kind);
        const rgb = layerRgbFromItem(item);
        if (role === "constraint") consCols.push(rgb);
        else densCols.push(rgb);
      }
      for (let i = 0; i < lastSceneBake.densLayers.length; i++) {
        if (densCols[i]) {
          lastSceneBake.densLayers[i].color = densCols[i].color;
          lastSceneBake.densLayers[i].color2 = densCols[i].color2;
          lastSceneBake.densLayers[i].colors = densCols[i].colors;
        }
      }
      for (let i = 0; i < lastSceneBake.constraints.length; i++) {
        if (consCols[i]) {
          lastSceneBake.constraints[i].color = consCols[i].color;
          lastSceneBake.constraints[i].color2 = consCols[i].color2;
          lastSceneBake.constraints[i].colors = consCols[i].colors;
        }
      }
      uploadSceneColors(lastSceneBake.densLayers.map((d) => d.colors || [d.color, d.color2]));
      if (isClipBakeGpuReady()) {
        uploadSceneVolumes({
          densLayers: lastSceneBake.densLayers,
          constraints: lastSceneBake.constraints,
          M: lastSceneBake.M,
        });
      }
      clipDirty = true;
    }
  },
  onStructuralChange: () => {
    scheduleUploadFit(0);
  },
});
exprListApi.render();
setExpressionsOnChange(() => {
  /* list mutations already call render from UI helpers */
});

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 0);
els.viewport.appendChild(renderer.domElement);

/** Axis labels on a higher canvas so isosurfaces (WebGPU or clip quad) never cover them. */
const labelRenderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
  powerPreference: "low-power",
});
labelRenderer.setPixelRatio(1);
labelRenderer.setClearColor(0x000000, 0);
labelRenderer.domElement.className = "axis-labels";
labelRenderer.domElement.style.cssText =
  "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2;";
els.viewport.appendChild(labelRenderer.domElement);
const labelScene = new THREE.Scene();

let themeColors = readThemeColors();
const lavaBg = createLavaBackground(themeColors);
applyClipGpuTheme(themeColors);

const scene = new THREE.Scene();
scene.add(lavaBg.mesh);
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
camera.position.set(3.2, 2.4, 4.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

/** @type {Float32Array | null} */
let worldCheb = null;
let fitDeg = 20;
let clipDirty = true;
let bakeMsSmooth = 0;
let lastDensSubmitMs = 0;
let densSubmittedThisFrame = false;
let frameDtSmooth = 16;
let lastRafAt = 0;
let lastVolumeM = 0;
let lastMetricsText = "";
let copyMetricsResetTimer = 0;
/** Last CPU Chebyshev→monomial fit breakdown (ms). */
let lastFitTiming = null;
let lastFitRel = NaN;
let lastNCoeff = 0;

const volPlaceholder = new Float32Array(8);
const volumeTex = new THREE.DataTexture(volPlaceholder, 2, 4, THREE.RedFormat, THREE.FloatType);
volumeTex.minFilter = THREE.LinearFilter;
volumeTex.magFilter = THREE.LinearFilter;
volumeTex.generateMipmaps = false;
volumeTex.flipY = false;
volumeTex.colorSpace = THREE.NoColorSpace;
volumeTex.needsUpdate = true;

/** @type {THREE.DataTexture | null} */
let clipVolumeTex = null;
let clipVolumeM = 0;

const clipUniforms = {
  uVolumeTex: { value: volumeTex },
  uGridM: { value: 2 },
  uFbW: { value: 1 },
  uFbH: { value: 1 },
  uHalf: { value: 2.5 },
  uScale: { value: 2.5 },
  uSteps: { value: 32 },
  uCameraPos: { value: new THREE.Vector3() },
  uDirM: { value: new THREE.Matrix3() },
  uAbsorbColor: { value: new THREE.Color(...themeColors.beerAbsorb) },
  uEmitColor: { value: new THREE.Color(...themeColors.beerEmit) },
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

const clipQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), clipMat);
clipQuad.frustumCulled = false;
clipQuad.visible = false;
scene.add(clipQuad);

const boxMat = new THREE.LineBasicMaterial({ color: themeColors.boxEdge });
const boxHelper = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(4, 4, 4)),
  boxMat,
);
scene.add(boxHelper);

/** World reference frame: unit grids + RGB axes (calculator-style). */
const worldGrid = new THREE.Group();
worldGrid.renderOrder = -1;
scene.add(worldGrid);
/** Axis letter sprites — own overlay canvas so volumes never cover them. */
const worldLabels = new THREE.Group();
labelScene.add(worldLabels);

function makeAxisLabel(text, color, position, labelStroke) {
  // High-res canvas so sprites stay sharp under orbit / retina.
  const css = 128;
  const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  const size = Math.round(css * dpr);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, size, size);
  ctx.scale(dpr, dpr);
  ctx.font = "600 72px 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Soft stroke reduces shimmer / jagged edges when minified by the GPU.
  ctx.lineWidth = 3;
  ctx.strokeStyle = labelStroke || "rgba(26, 18, 40, 0.58)";
  ctx.strokeText(text, css / 2, css / 2 + 1);
  ctx.fillStyle = color;
  ctx.fillText(text, css / 2, css / 2 + 1);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.42, 0.42, 0.42);
  spr.position.copy(position);
  spr.renderOrder = 10;
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
  while (worldLabels.children.length) {
    const child = worldLabels.children.pop();
    child.material?.map?.dispose?.();
    child.material?.dispose?.();
  }

  const h = Math.max(0.5, half);
  // Grid a bit past the fit box; aim for ~1 world-unit cells.
  const extent = Math.ceil(h + 0.5);
  const size = extent * 2;
  const divisions = Math.max(2, size);
  const tc = readThemeColors();
  const major = tc.gridMajor;
  const minor = tc.gridMinor;

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
    ...tc.axisXRgb, ...tc.axisXRgb,
    ...tc.axisYRgb, ...tc.axisYRgb,
    ...tc.axisZRgb, ...tc.axisZRgb,
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
  worldLabels.add(makeAxisLabel("x", tc.axisX, new THREE.Vector3(tip, 0, 0), tc.labelStroke));
  worldLabels.add(makeAxisLabel("y", tc.axisY, new THREE.Vector3(0, tip, 0), tc.labelStroke));
  worldLabels.add(makeAxisLabel("z", tc.axisZ, new THREE.Vector3(0, 0, tip), tc.labelStroke));

  // WebGPU path draws the same grid against iso depth (no texture copy).
  syncClipGpuWorldGrid(h);
}

/** Fit / march use half-extent h; UI “box size” is full edge length S = 2h. */
function setBoxSize(size) {
  const s = Math.max(1e-6, size);
  const h = 0.5 * s;
  boxHelper.geometry.dispose();
  boxHelper.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(s, s, s));
  clipUniforms.uHalf.value = h;
  rebuildWorldGrid(h);
}

rebuildWorldGrid(2);

function applyThemeToScene() {
  themeColors = readThemeColors();
  lavaBg.setColors(themeColors);
  boxMat.color.setHex(themeColors.boxEdge);
  clipUniforms.uAbsorbColor.value.setRGB(...themeColors.beerAbsorb);
  clipUniforms.uEmitColor.value.setRGB(...themeColors.beerEmit);
  applyClipGpuTheme(themeColors);
  rebuildWorldGrid(clipUniforms.uHalf.value);
  clipDirty = true;
}

onThemeChange((_resolved, pref) => {
  if (els.themePref && els.themePref.value !== pref) els.themePref.value = pref;
});

if (els.themePref) {
  els.themePref.value = getThemePref();
  els.themePref.addEventListener("change", () => {
    setThemePref(/** @type {import("./theme.js").ThemePref} */ (els.themePref.value));
  });
}
onThemeChange(() => applyThemeToScene());

function fmtRel(v) {
  if (!Number.isFinite(v)) return "∞";
  if (v < 1e-3) return v.toExponential(2);
  return v.toPrecision(3);
}

function setErr(msg) {
  els.err.textContent = msg || "";
}

/** Highlight expression fields when compile fails (preserve duplicate-var warnings). */
function setExprCompileOk(ok) {
  els.exprList?.querySelectorAll(".expr-field").forEach((mf) => {
    const row = mf.closest?.(".expr-row");
    const id = row instanceof HTMLElement ? row.dataset.id : null;
    const warn = id ? getExprWarning(id) : null;
    if (warn) {
      mf.classList.add("invalid");
      if (mf instanceof HTMLElement) mf.title = warn;
      return;
    }
    mf.classList.toggle("invalid", !ok);
    if (mf instanceof HTMLElement && !ok) {
      /* keep existing title if any */
    } else if (mf instanceof HTMLElement) {
      mf.removeAttribute("title");
    }
  });
}

function syncExprCompileState() {
  try {
    const result = compileAllExprs({ rebuildUi: true });
    setExprCompileOk(true);
    const warn = result?.warnings?.length ? result.warnings.join(" · ") : "";
    setErr(warn);
    return true;
  } catch (e) {
    setExprCompileOk(false);
    setErr(e instanceof Error ? e.message : String(e));
    return false;
  }
}

function readIsoLevel() {
  return Number.isFinite(lastExprMeta.isoLevel) ? lastExprMeta.isoLevel : 0;
}

let loopFps = 0;
let loopFpsFrames = 0;
let loopFpsLast = performance.now();
let cpuMsSmooth = 0;

/** FPS shown in HUD — GPU present rate when WebGPU is active, else rAF spacing. */
function hudFpsText() {
  const loop = 1000 / Math.max(1, frameDtSmooth);
  if (useGpuClipPath()) {
    const p = getClipGpuProfile();
    const presentFresh =
      p.presentIntervalMs > 0 && performance.now() - p.lastPresentAt < 1200;
    if (presentFresh) {
      const gpu = 1000 / p.presentIntervalMs;
      if (loop > gpu * 1.12) {
        return `${Math.round(gpu)} fps · loop ${Math.round(loop)}`;
      }
      return `${Math.round(gpu)} fps`;
    }
  }
  return `${Math.round(loop)} fps`;
}

function hudText() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const pr = renderer.getPixelRatio();
  const submit = densSubmittedThisFrame
    ? `submit ${bakeMsSmooth.toFixed(0)}ms`
    : `submit miss (last ${lastDensSubmitMs.toFixed(0)}ms)`;
  const p = getClipGpuProfile();
  const gpuSplit = p.timestamps
    ? ` · idct ${p.idctMs.toFixed(2)}/march ${p.marchMs.toFixed(1)}`
    : "";
  const present =
    p.presentIntervalMs > 0 && performance.now() - p.lastPresentAt < 1200
      ? ` · present ${p.presentIntervalMs.toFixed(0)}ms`
      : "";
  const clip = ` · rAF ${frameDtSmooth.toFixed(0)}ms · ${submit}${gpuSplit}${present} · vol ${lastVolumeM}³`;
  return `clip-grid · ${hudFpsText()} · ${cpuMsSmooth.toFixed(1)}ms js${clip} · ${Math.round(w * pr)}×${Math.round(h * pr)}`;
}

function buildMetricsReport() {
  const fbW = Math.max(1, renderer.domElement.width);
  const fbH = Math.max(1, renderer.domElement.height);
  const p = getClipGpuProfile();
  const lines = [
    `poly-cloud metrics  ${new Date().toISOString()}`,
    `deg             ${fitDeg}`,
    `scale           ${clipUniforms.uScale.value}`,
    `steps           ${clipUniforms.uSteps.value}`,
    `box_size        ${2 * clipUniforms.uHalf.value}`,
    `march_downscale ${marchDownscale()}×`,
    `march_resolution ${(100 / marchDownscale()).toFixed(1)}%`,
    `viewport        ${fbW}×${fbH}`,
    `march_fb_req    ${marchFramebufferSize().mw}×${marchFramebufferSize().mh}`,
    `gpu_march_fb    ${p.marchFbW && p.marchFbH ? `${p.marchFbW}×${p.marchFbH}` : "—"}`,
    `loop_fps        ${Math.round(loopFps)}`,
    `loop_ms         ${frameDtSmooth.toFixed(2)}`,
    `js_frame_ms     ${cpuMsSmooth.toFixed(2)}`,
    `gpu_path        ${useGpuClipPath() ? "webgpu" : "cpu/webgl"}`,
    `gpu_method      ${p.method || "—"}`,
    `expr_kind       ${lastExprMeta.kind}`,
    `shade           ${lastExprMeta.shade}`,
    `iso_level       ${readIsoLevel()}`,
    `grid_m          ${lastVolumeM || p.gridM || "—"}`,
    `march_submit_ms ${densSubmittedThisFrame ? bakeMsSmooth.toFixed(2) : "—"}`,
    `march_last_ms   ${lastDensSubmitMs.toFixed(2)}`,
    `gpu_timestamps  ${p.timestamps ? "yes" : "no"}`,
    `idct_ms         ${p.idctMs ? p.idctMs.toFixed(3) : "n/a"}`,
    `gpu_march_ms    ${p.timestamps ? p.marchMs.toFixed(3) : "n/a"}`,
    `gpu_present_ms  ${p.presentWallMs > 0 ? p.presentWallMs.toFixed(2) : "n/a"}`,
    `gpu_present_iv  ${p.presentIntervalMs > 0 ? p.presentIntervalMs.toFixed(2) : "n/a"}`,
    `gpu_present_fps ${
      p.presentIntervalMs > 0 ? Math.round(1000 / p.presentIntervalMs) : "n/a"
    }`,
    `fit_rel_L2      ${Number.isFinite(lastFitRel) ? fmtRel(lastFitRel) : "—"}`,
    `n_coeffs        ${lastNCoeff || "—"}`,
  ];
  if (lastFitTiming) {
    const t = lastFitTiming;
    lines.push(
      `fit_total_ms    ${t.totalMs.toFixed(2)}`,
      `fit_sample_ms   ${t.sampleMs.toFixed(2)}`,
      `fit_cheb_ms     ${t.chebMs.toFixed(2)}`,
      `fit_mono_ms     ${t.monoMs.toFixed(2)}`,
      `fit_l2_ms       ${t.l2Ms.toFixed(2)}`,
      `fit_upload_ms   ${t.uploadMs.toFixed(2)}`,
    );
    if (Number.isFinite(t.fittedCount)) {
      lines.push(`fit_layers      ${t.fittedCount}`);
    }
    if (Number.isFinite(t.keyframedCount)) {
      lines.push(`kf_layers       ${t.keyframedCount}`);
    }
    if (Number.isFinite(t.kfBakeMs) && t.kfBakeMs > 0) {
      lines.push(`kf_bake_ms      ${t.kfBakeMs.toFixed(2)}`);
    }
    if (Number.isFinite(t.kfLerpMs) && t.kfLerpMs > 0) {
      lines.push(`kf_lerp_ms      ${t.kfLerpMs.toFixed(2)}`);
    } else if (Number.isFinite(t.keyframedCount) && t.keyframedCount > 0) {
      lines.push(`kf_blend        gpu`);
    }
    if (Number.isFinite(t.kfK) && t.keyframedCount > 0) {
      lines.push(`kf_K            ${t.kfK}`);
    }
  }
  const pv = getParamValues();
  const pNames = Object.keys(pv);
  if (pNames.length) {
    lines.push(
      `params          ${pNames.map((n) => `${n}=${fmtParamNum(pv[n])}`).join(" ")}`,
    );
  }
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

function viewportSize() {
  const vw = els.viewport.clientWidth;
  const vh = Math.max(els.viewport.clientHeight, 1);
  return { vw, vh };
}

/** CSS px covered by the floating sidebar (0 on narrow layouts). */
function compositionCoveredWidth(vw) {
  if (typeof window !== "undefined" && window.matchMedia("(max-width: 800px)").matches) {
    return 0;
  }
  const cs = getComputedStyle(document.documentElement);
  const inset = parseFloat(cs.getPropertyValue("--panel-inset"));
  const pw = parseFloat(cs.getPropertyValue("--panel-w"));
  const covered = (Number.isFinite(inset) ? inset : 12) + (Number.isFinite(pw) ? pw : 360);
  return Math.min(Math.max(0, covered), Math.max(0, vw - 160));
}

/** NDC x of the free-region center (0 when the panel does not inset composition). */
function compositionNdcOffsetX(vw) {
  const covered = compositionCoveredWidth(vw);
  if (covered <= 1 || vw <= covered + 40) return 0;
  return covered / vw;
}

function applyCameraComposition(vw, vh) {
  // Keep projection in sync with offsetDirMatrix used by volume rays:
  // rays aim forward at NDC x = +offset (free-region center to the right of the panel),
  // so world points on the view axis must project to that same NDC x.
  if (typeof camera.clearViewOffset === "function") camera.clearViewOffset();
  camera.aspect = vw / Math.max(vh, 1);
  camera.updateProjectionMatrix();
  const o = compositionNdcOffsetX(vw);
  if (Math.abs(o) > 1e-12) {
    const e = camera.projectionMatrix.elements;
    // Left-multiply by translate(x' = x + o*w): column c, row 0 += o * row 3.
    for (let c = 0; c < 4; c++) {
      e[c * 4 + 0] += o * e[c * 4 + 3];
    }
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
}

/** Clip-grid Beer march internal resolution (CSS-upscaled to the viewport). */
function marchResolutionScale() {
  return 1 / marchDownscale();
}

function marchFramebufferSize() {
  const { vw, vh } = viewportSize();
  const s = marchResolutionScale();
  return {
    mw: Math.max(1, Math.round(vw * s)),
    mh: Math.max(1, Math.round(vh * s)),
  };
}

function applyDisplaySize(rw, rh, vw, vh, { markClipDirty = true } = {}) {
  applyCameraComposition(vw, vh);
  renderer.setSize(rw, rh, false);
  const canvas = renderer.domElement;
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.position = "absolute";
  canvas.style.inset = "0";
  labelRenderer.setSize(rw, rh, false);
  const labelCanvas = labelRenderer.domElement;
  labelCanvas.style.width = "100%";
  labelCanvas.style.height = "100%";
  if (markClipDirty) clipDirty = true;
  if (els.hud) els.hud.textContent = hudText();
}

function resize() {
  const { vw, vh } = viewportSize();
  applyDisplaySize(vw, vh, vw, vh, { markClipDirty: true });
  // Keep the WebGPU overlay matched to the viewport CSS box immediately
  // (buffer stays at march resolution; style fills the viewport).
  const { mw, mh } = marchFramebufferSize();
  resizeClipGpuCanvas(mw, mh);
}

/** @type {{ densLayers: any[], constraints: any[], M: number, dens: Float32Array | null } | null} */
let lastSceneBake = null;

/** Fit-time: IDCT each expression → GPU scene (manifolds + densities). */
function bakeChebVolume() {
  if (!lastSceneBake) return null;
  const { densLayers, constraints, M, dens } = lastSceneBake;
  if (isClipBakeGpuReady()) {
    const up = uploadSceneVolumes({ densLayers, constraints, M });
    if (up) bakeMsSmooth = bakeMsSmooth * 0.5 + up.bakeMs * 0.5;
  }
  lastVolumeM = M;
  if (dens) applyVolumeTexture(dens, M);
  return { dens, M };
}

function applyVolumeTexture(dens, M) {
  const h = M * M;
  if (!clipVolumeTex || clipVolumeM !== M) {
    if (clipVolumeTex) clipVolumeTex.dispose();
    clipVolumeTex = new THREE.DataTexture(dens, M, h, THREE.RedFormat, THREE.FloatType);
    clipVolumeTex.minFilter = THREE.LinearFilter;
    clipVolumeTex.magFilter = THREE.LinearFilter;
    clipVolumeTex.generateMipmaps = false;
    clipVolumeTex.flipY = false;
    clipVolumeTex.colorSpace = THREE.NoColorSpace;
    clipVolumeTex.needsUpdate = true;
    clipVolumeM = M;
    clipUniforms.uVolumeTex.value = clipVolumeTex;
  } else {
    clipVolumeTex.image.data.set(dens);
    clipVolumeTex.needsUpdate = true;
  }
  clipUniforms.uGridM.value = M;
}

function syncClipFiberUniforms() {
  // CPU/WebGL path draws into the full Three.js canvas — NDC must use that
  // buffer size, not the march-downscale size (that is GPU-canvas only).
  camera.updateMatrixWorld(true);
  const { vw } = viewportSize();
  const { sx, sy } = perspectiveDirScale(camera);
  const M = offsetDirMatrix(ndcToDirMatrix(camera, sx, sy), compositionNdcOffsetX(vw));
  clipUniforms.uFbW.value = Math.max(1, renderer.domElement.width);
  clipUniforms.uFbH.value = Math.max(1, renderer.domElement.height);
  clipUniforms.uDirM.value.set(M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]);
  clipUniforms.uCameraPos.value.copy(camera.position);
}

function useGpuClipPath() {
  return isClipBakeGpuReady() && isClipMarchReady();
}

function syncClipPresentation() {
  const hasVolume = hasUploadedVolume() || Boolean(
    lastSceneBake && (lastSceneBake.densLayers.length || lastSceneBake.constraints.length),
  );
  const gpu = useGpuClipPath() && hasVolume;
  clipQuad.visible = !gpu && hasVolume && Boolean(worldCheb);
  // Grid/box depth-test on the WebGPU overlay; axis labels stay on a
  // higher WebGL canvas so isosurfaces never cover them.
  worldGrid.visible = !gpu;
  boxHelper.visible = !gpu;
  setClipGpuCanvasVisible(isClipBakeGpuReady());
}

/** Per-frame GPU volume march (IDCT bake is fit-time only). */
function drawClipGpuFrame() {
  densSubmittedThisFrame = false;
  const { mw, mh } = marchFramebufferSize();
  if (!lastSceneBake || !isClipBakeGpuReady()) return false;
  if (!hasUploadedVolume()) {
    clearClipGpuFrame(mw, mh);
    clipDirty = false;
    return true;
  }
  if (!useGpuClipPath()) return false;

  camera.updateMatrixWorld(true);
  const { vw } = viewportSize();
  const t0 = performance.now();
  const ok = renderClipFrameGpu({
    camera,
    half: clipUniforms.uHalf.value,
    fbW: mw,
    fbH: mh,
    scale: clipUniforms.uScale.value,
    steps: clipUniforms.uSteps.value | 0,
    ndcOffsetX: compositionNdcOffsetX(vw),
  });
  const submitMs = performance.now() - t0;
  if (ok) {
    densSubmittedThisFrame = true;
    lastDensSubmitMs = submitMs;
    bakeMsSmooth = bakeMsSmooth * 0.85 + submitMs * 0.15;
    clipDirty = false;
  }
  return ok;
}

function syncClipCpuVolume() {
  if (useGpuClipPath()) return;
  if (!worldCheb || !hasUploadedVolume()) {
    clipQuad.visible = false;
    return;
  }
  if (clipDirty || !clipVolumeTex) bakeChebVolume();
  syncClipFiberUniforms();
  setClipGpuCanvasVisible(false);
  clipQuad.visible = true;
  clipDirty = false;
}

async function prepareClipGpuForDegree(deg) {
  try {
    const ok = await initClipBakeGpu(els.viewport);
    if (!ok) return false;
    await ensurePipelinesForDegree(deg);
    if (lastSceneBake) bakeChebVolume();
    syncClipPresentation();
    return true;
  } catch (e) {
    console.warn("[clip-grid] pipeline specialize failed", e);
    return false;
  }
}

function uploadFit(opts = {}) {
  const fromAnim = !!opts.fromAnim;
  setErr("");
  try {
    const boxSize = Number(els.boxSize.value);
    const deg = Number(els.deg.value);
    const densScale = Number(els.scale.value);
    const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
    els.steps.value = String(steps);
    if (!(boxSize > 0)) throw new Error("box size must be > 0");
    if (deg < 1 || deg > MAX_DEG) throw new Error(`poly deg must be 1…${MAX_DEG}`);
    const half = 0.5 * boxSize;

    const tUpload = performance.now();
    const { layers } = compileAllExprs({ rebuildUi: false });
    setExprCompileOk(true);

    // No visible / non-empty expressions → clear volume, draw nothing.
    if (!layers.length) {
      clearKeyframeCaches();
      lastSceneBake = { densLayers: [], constraints: [], M: Math.max(2, deg + 1), dens: null };
      lastFitTiming = null;
      lastNCoeff = 0;
      lastFitRel = NaN;
      worldCheb = null;
      fitDeg = deg;
      if (isClipBakeGpuReady()) {
        uploadSceneVolumes({ densLayers: [], constraints: [], M: lastSceneBake.M });
      }
      clipUniforms.uScale.value = densScale;
      clipUniforms.uSteps.value = steps;
      setBoxSize(boxSize);
      clipQuad.visible = false;
      clipDirty = true;
      if (!fromAnim) resize();
      syncClipPresentation();
      if (isClipBakeGpuReady()) {
        const { mw, mh } = marchFramebufferSize();
        clearClipGpuFrame(mw, mh);
      }
      return;
    }

    const densLayers = [];
    const constraints = [];
    let cheb = null;
    let fitRel = NaN;
    let timingAcc = { sampleMs: 0, chebMs: 0, monoMs: 0, l2Ms: 0, totalMs: 0 };
    let M = deg + 1;
    let fittedCount = 0;
    let keyframedCount = 0;
    let keyframeBaked = false;
    let densKeyframedCpu = false;

    // Anim ticks: only refit layers that depend on dirty params; reuse the rest.
    // Dirty layers with exactly one animating slider: GPU keyframe blend (iso) / CPU lerp (dens).
    const dirty = fromAnim ? collectAnimDirtyParams() : null;
    if (fromAnim) beginKeyframePass();
    else clearKeyframeCaches();

    /** @type {Map<string, any>} */
    const prevById = new Map();
    const canReuseCache =
      fromAnim &&
      dirty &&
      lastSceneBake &&
      lastSceneBake.deg === deg &&
      Math.abs((lastSceneBake.half ?? NaN) - half) < 1e-12;
    if (canReuseCache) {
      for (const d of lastSceneBake.densLayers) {
        if (d.id) prevById.set(d.id, { kind: "density", ...d });
      }
      for (const c of lastSceneBake.constraints) {
        if (c.id) prevById.set(c.id, { kind: "constraint", ...c });
      }
    }

    const baseParams = getParamValues();

    for (const L of layers) {
      const { color, color2, colors } = layerRgbFromItem(L.item);
      const depends =
        !dirty ||
        L.compiled.freeParams.some((p) => dirty.has(p));
      const prev = canReuseCache && !depends ? prevById.get(L.item.id) : null;
      const prevHasKf =
        prev && Array.isArray(prev.keyframes) && prev.keyframes.length > 0;
      const reuseDens =
        prev &&
        prev.kind === (L.role === "constraint" ? "constraint" : "density") &&
        (prev.dens instanceof Float32Array || prevHasKf);

      if (reuseDens) {
        if (prevHasKf) {
          M = Math.round(Math.cbrt(prev.keyframes[0].dens.length)) || M;
        } else {
          M = Math.round(Math.cbrt(prev.dens.length)) || M;
        }
        if (L.role === "constraint") {
          if (prevHasKf) {
            constraints.push({
              id: L.item.id,
              keyframes: prev.keyframes,
              blend: prev.blend || { i0: 0, i1: 0, t: 0 },
              color,
              color2,
              colors,
              isoLevel: L.compiled.isoLevel ?? prev.isoLevel ?? 0,
              cheb: prev.cheb,
              fitRel: prev.fitRel,
            });
          } else {
            constraints.push({
              id: L.item.id,
              dens: prev.dens,
              gx: prev.gx,
              gy: prev.gy,
              gz: prev.gz,
              color,
              color2,
              colors,
              isoLevel: L.compiled.isoLevel ?? prev.isoLevel ?? 0,
            });
          }
        } else {
          densLayers.push({ id: L.item.id, dens: prev.dens, color, color2 });
        }
        if (!cheb && prev.cheb) {
          cheb = prev.cheb;
          fitRel = prev.fitRel ?? fitRel;
        }
        continue;
      }

      // Keyframe path: one dirty cosine-animated slider → GPU blend (iso) / CPU lerp (dens).
      const kfParam =
        fromAnim && depends
          ? keyframeAnimParam(L.compiled.freeParams, dirty)
          : null;
      if (kfParam) {
        noteKeyframeLayer();
        keyframedCount++;
        if (L.role === "constraint") {
          const sample = ensureLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "constraint",
            isoLevel: L.compiled.isoLevel ?? 0,
            paramName: kfParam,
            compiled: L.compiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          constraints.push({
            id: L.item.id,
            keyframes: sample.frames,
            blend: sample.blend,
            color,
            color2,
            colors,
            isoLevel: L.compiled.isoLevel ?? 0,
            cheb: sample.cheb,
            fitRel: sample.fitRel,
          });
          if (!cheb && sample.cheb) {
            cheb = sample.cheb;
            fitRel = sample.fitRel ?? fitRel;
          }
        } else {
          const sample = sampleLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "density",
            isoLevel: L.compiled.isoLevel ?? 0,
            paramName: kfParam,
            compiled: L.compiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          densLayers.push({
            id: L.item.id,
            dens: sample.dens.slice(),
            color,
            color2,
            colors,
            cheb: sample.cheb,
            fitRel: sample.fitRel,
          });
          densKeyframedCpu = true;
          if (!cheb && sample.cheb) {
            cheb = sample.cheb;
            fitRel = sample.fitRel ?? fitRel;
          }
        }
        continue;
      }

      const fit = fitChebyshev3D(L.fn, half, deg, {
        skipL2: fromAnim || layers.length > 1,
      });
      fittedCount++;
      const idct = idctCheb3D(fit.cheb, fit.deg, fit.deg + 1);
      M = idct.M;
      if (L.role === "constraint") {
        const grad = idctChebGrad3D(fit.cheb, fit.deg, fit.deg + 1);
        constraints.push({
          id: L.item.id,
          dens: idct.dens,
          gx: grad.gx,
          gy: grad.gy,
          gz: grad.gz,
          color,
          color2,
          colors,
          isoLevel: L.compiled.isoLevel ?? 0,
          cheb: fit.cheb,
          fitRel: fit.fitRelL2,
        });
      } else {
        densLayers.push({
          id: L.item.id,
          dens: idct.dens,
          color,
          color2,
          colors,
          cheb: fit.cheb,
          fitRel: fit.fitRelL2,
        });
      }
      if (!cheb) {
        cheb = fit.cheb;
        fitRel = fit.fitRelL2;
      }
      for (const k of Object.keys(timingAcc)) {
        timingAcc[k] += fit.timing?.[k] || 0;
      }
    }

    // Preview texture: sum of density layers only (never constraints).
    let densSum = null;
    if (densLayers.length) {
      densSum = new Float32Array(M * M * M);
      for (const d of densLayers) {
        for (let i = 0; i < densSum.length; i++) densSum[i] += d.dens[i] || 0;
      }
    }

    lastSceneBake = {
      densLayers,
      constraints,
      M,
      dens: densSum,
      deg,
      half,
      fittedCount,
      keyframedCount,
    };
    const uploadMs = performance.now() - tUpload;
    const kf = getKeyframeMetrics();
    lastFitTiming = {
      ...timingAcc,
      uploadMs,
      fittedCount,
      keyframedCount,
      kfBakeMs: kf.bakeMs,
      kfLerpMs: kf.lerpMs,
      kfK: kf.K,
    };
    lastNCoeff = (deg + 1) ** 3 * layers.length;
    if (Number.isFinite(fitRel)) lastFitRel = fitRel;

    if (cheb) worldCheb = cheb;
    else if (!fromAnim) worldCheb = null;
    fitDeg = deg;
    // GPU iso keyframes: upload only on bake / dens CPU lerp / full fit.
    // Warm anim ticks only update blend uniforms.
    const needUpload =
      fittedCount > 0 || keyframeBaked || densKeyframedCpu || !fromAnim;
    if (needUpload) {
      bakeChebVolume();
    } else if (keyframedCount > 0) {
      setConstraintKeyframeBlends(
        constraints
          .filter((c) => c.blend && c.id != null)
          .map((c) => ({ id: c.id, i0: c.blend.i0, i1: c.blend.i1, t: c.blend.t })),
      );
    }

    clipUniforms.uScale.value = densScale;
    clipUniforms.uSteps.value = steps;
    setBoxSize(boxSize);

    if (!fromAnim) resize();
    clipDirty = true;
    if (needUpload) {
      void prepareClipGpuForDegree(deg).then(() => {
        if (lastSceneBake) bakeChebVolume();
        syncClipPresentation();
        if (!useGpuClipPath()) {
          clipDirty = true;
          syncClipCpuVolume();
        }
      });
    } else {
      syncClipPresentation();
    }
  } catch (e) {
    try {
      compileAllExprs();
      setExprCompileOk(true);
    } catch {
      setExprCompileOk(false);
    }
    setErr(e instanceof Error ? e.message : String(e));
  }
}

/** Debounced Chebyshev refit when expr / deg / box size become valid. */
let fitTimer = 0;
const FIT_DEBOUNCE_MS = 320;
/** Min ms between full anim refits (DCT / dens CPU lerp). GPU blends update every frame. */
const ANIM_FIT_MIN_MS = 50;
let lastAnimFitAt = 0;

/** Push current param → keyframe blend uniforms at full RAF rate (no volume rewrite). */
function tickGpuKeyframeBlends() {
  const cons = lastSceneBake?.constraints;
  if (!cons?.length || !isClipBakeGpuReady()) return false;
  /** @type {{ id: string, i0: number, i1: number, t: number }[]} */
  const blends = [];
  for (const c of cons) {
    if (!c?.id || !Array.isArray(c.keyframes) || !c.keyframes.length) continue;
    const b = peekKeyframeBlend(c.id);
    if (!b) continue;
    c.blend = { i0: b.i0, i1: b.i1, t: b.t };
    blends.push(b);
  }
  if (!blends.length) return false;
  setConstraintKeyframeBlends(blends);
  clipDirty = true;
  return true;
}

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
  clipUniforms.uScale.value = densScale;
  clipUniforms.uSteps.value = steps;
  clipDirty = true;
}

els.preset.addEventListener("change", () => {
  applyPreset(els.preset.value);
  if (fitTimer) clearTimeout(fitTimer);
  uploadFit();
});
els.deg.addEventListener("input", () => scheduleUploadFit(200));
els.deg.addEventListener("change", () => scheduleUploadFit(0));
els.boxSize.addEventListener("input", () => scheduleUploadFit(200));
els.boxSize.addEventListener("change", () => scheduleUploadFit(0));
els.scale.addEventListener("input", applyRenderHyperparams);
els.scale.addEventListener("change", applyRenderHyperparams);
els.steps.addEventListener("input", applyRenderHyperparams);
els.steps.addEventListener("change", applyRenderHyperparams);
function markMarchDirty() {
  syncMarchSlider();
  resetClipGpuProfile();
  clipDirty = true;
}
els.marchDownscale?.addEventListener("input", markMarchDirty);
els.marchDownscale?.addEventListener("change", markMarchDirty);
els.reset.addEventListener("click", () => {
  camera.position.set(3.2, 2.4, 4.2);
  controls.target.set(0, 0, 0);
  controls.update();
  clipDirty = true;
});
els.copyMetrics?.addEventListener("click", (ev) => {
  ev.preventDefault();
  ev.stopPropagation();
  void copyMetricsToClipboard();
});

function openSettingsDialog() {
  if (!els.settingsDialog || typeof els.settingsDialog.showModal !== "function") return;
  if (!els.settingsDialog.open) els.settingsDialog.showModal();
}
function closeSettingsDialog() {
  if (!els.settingsDialog?.open) return;
  els.settingsDialog.close();
}
els.openSettings?.addEventListener("click", () => openSettingsDialog());
els.closeSettings?.addEventListener("click", () => closeSettingsDialog());
els.settingsDialog?.addEventListener("click", (ev) => {
  if (ev.target === els.settingsDialog) closeSettingsDialog();
});

window.addEventListener("resize", resize);
resize();

/** Drag the sidebar edge to change --panel-w; persists in localStorage. */
(function initPanelResize() {
  const handle = document.getElementById("panelResize");
  if (!handle) return;
  const PANEL_MIN = 240;
  const PANEL_MAX = 720;
  const STORAGE_KEY = "poly-cloud-panel-w";

  function panelInset() {
    const raw = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-inset"));
    return Number.isFinite(raw) ? raw : 12;
  }

  function clampW(w) {
    const max = Math.min(PANEL_MAX, Math.max(PANEL_MIN, window.innerWidth - 2 * panelInset() - 160));
    return Math.round(Math.min(max, Math.max(PANEL_MIN, w)));
  }

  function applyW(w) {
    const px = clampW(w);
    document.documentElement.style.setProperty("--panel-w", `${px}px`);
    handle.setAttribute("aria-valuenow", String(px));
    return px;
  }

  try {
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    if (Number.isFinite(saved) && saved > 0) applyW(saved);
  } catch (_) {
    /* ignore */
  }

  handle.setAttribute("aria-valuemin", String(PANEL_MIN));
  handle.setAttribute("aria-valuemax", String(PANEL_MAX));

  let dragging = false;

  function onMove(ev) {
    if (!dragging) return;
    const x = ev.touches ? ev.touches[0].clientX : ev.clientX;
    applyW(x - panelInset());
    resize();
  }

  function onUp() {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.classList.remove("panel-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
    try {
      const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w"));
      if (Number.isFinite(cur)) localStorage.setItem(STORAGE_KEY, String(cur));
    } catch (_) {
      /* ignore */
    }
    resize();
  }

  handle.addEventListener("pointerdown", (ev) => {
    if (window.matchMedia("(max-width: 800px)").matches) return;
    ev.preventDefault();
    dragging = true;
    handle.classList.add("dragging");
    document.body.classList.add("panel-resizing");
    handle.setPointerCapture?.(ev.pointerId);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });

  handle.addEventListener("keydown", (ev) => {
    const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w")) || 360;
    const step = ev.shiftKey ? 40 : 16;
    if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      applyW(cur - step);
      resize();
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      applyW(cur + step);
      resize();
    }
  });
})();

if (els.hud) els.hud.textContent = "clip-grid · idct volume";
uploadFit();

controls.addEventListener("start", () => {
  clipDirty = true;
});

controls.addEventListener("change", () => {
  clipDirty = true;
});

controls.addEventListener("end", () => {
  clipDirty = true;
});

function frame(rafNow) {
  const t0 = performance.now();
  if (lastRafAt > 0) {
    const rafDt = rafNow > 0 ? rafNow - lastRafAt : t0 - lastRafAt;
    if (rafDt > 0 && rafDt < 500) {
      frameDtSmooth = frameDtSmooth * 0.85 + rafDt * 0.15;
    }
  }
  lastRafAt = rafNow > 0 ? rafNow : t0;
  loopFpsFrames++;

  // Named param animation / LaTeX time → Chebyshev refit (throttled).
  if (anyParamNeedsTick()) {
    const tSec = t0 / 1000;
    const animChanged = tickParamAnimation(tSec);
    const eqChanged = evalParamEquations(tSec);
    if (animChanged || eqChanged) {
      if (animChanged) {
        for (const name of listParamNames()) {
          const p = getParam(name);
          if (p?.exprId && !p.driven) {
            updateExprSilent(p.exprId, {
              latex: p.latex,
              sliderAnimating: p.animating,
              sliderPhase: p.phase,
            });
          }
        }
      }
      exprListApi?.syncAllParamSliders?.();
      // GPU iso keyframes: blend every frame so the thumb + field stay continuous.
      tickGpuKeyframeBlends();
      if (t0 - lastAnimFitAt >= ANIM_FIT_MIN_MS) {
        lastAnimFitAt = t0;
        if (fitTimer) {
          clearTimeout(fitTimer);
          fitTimer = 0;
        }
        uploadFit({ fromAnim: true });
      }
    }
  }

  if (t0 - loopFpsLast >= 500) {
    const winMs = t0 - loopFpsLast;
    loopFps = (loopFpsFrames * 1000) / winMs;
    loopFpsFrames = 0;
    loopFpsLast = t0;
    if (els.hud) els.hud.textContent = hudText();
    refreshMetricsDump();
  }

  controls.update();
  clipUniforms.uCameraPos.value.copy(camera.position);

  if (!useGpuClipPath()) {
    syncClipCpuVolume();
  }

  lavaBg.setTime(t0 / 1000);
  lavaBg.syncCamera(camera);
  renderer.autoClear = true;
  renderer.render(scene, camera);

  if (isClipBakeGpuReady()) {
    if (hasUploadedVolume() && useGpuClipPath()) {
      drawClipGpuFrame();
    } else if (!hasUploadedVolume()) {
      const { mw, mh } = marchFramebufferSize();
      clearClipGpuFrame(mw, mh);
      densSubmittedThisFrame = false;
    }
  }

  labelRenderer.render(labelScene, camera);

  const dt = performance.now() - t0;
  cpuMsSmooth = cpuMsSmooth * 0.85 + dt * 0.15;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

void initClipBakeGpu(els.viewport).then(async (ok) => {
  if (ok && worldCheb) await prepareClipGpuForDegree(fitDeg);
  syncClipPresentation();
  clipDirty = true;
  if (!useGpuClipPath()) {
    syncClipCpuVolume();
  }
});
