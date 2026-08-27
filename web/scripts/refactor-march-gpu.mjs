import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const marchPath = path.join(__dirname, "../src/render/webgpu/march.js");
let src = fs.readFileSync(marchPath, "utf8");

const stateVars = [
  "device", "ctx", "canvas", "canvasFormat", "isoPipeline", "beerPipeline",
  "fxaaPipeline", "ssaoPipeline", "gridPipeline", "gridParamBuf", "gridVertexBuf",
  "gridVertexCapacity", "gridVertexCount", "gridHalf", "labelPipeline", "labelVertexBuf",
  "labelAtlasTex", "labelAtlasSamp", "labelAtlasDirty", "drawParamBuf", "drawParamBufBeer",
  "fxaaParamBuf", "ssaoParamBuf", "volumeBuf", "volumeCapacity", "colorBuf",
  "occlTex", "occlW", "occlH", "depthTex", "depthW", "depthH", "normalTex", "normalW", "normalH",
  "sceneColorTex", "sceneColorW", "sceneColorH", "sceneColorAoTex", "sceneColorAoW", "sceneColorAoH",
  "fxaaSampler", "sceneConstraints", "densPacked", "densGradStops", "densLayerCount", "densBase",
  "sceneM", "sceneEpoch", "scenePacked", "initFailed", "initPromise", "timestampsSupported",
  "stampQuerySet", "stampResolveBuf", "stampReadBuf", "stampReadPending", "profileBakeMs",
  "profileMarchMs", "profileMarchFbW", "profileMarchFbH", "profilePresentWallMs",
  "profilePresentIntervalMs", "lastPresentAt", "profileMethod", "profileGridM", "builtEpoch",
  "isoInterpHermite",
];

function removeFunction(src, name) {
  const start = src.indexOf(`function ${name}(`);
  if (start === -1) {
    const asyncStart = src.indexOf(`async function ${name}(`);
    if (asyncStart === -1) return src;
    return removeBlockFrom(src, asyncStart);
  }
  return removeBlockFrom(src, start);
}

function removeExportFunction(src, name) {
  const start = src.indexOf(`export function ${name}(`);
  if (start === -1) {
    const asyncStart = src.indexOf(`export async function ${name}(`);
    if (asyncStart === -1) return src;
    return removeBlockFrom(src, asyncStart);
  }
  return removeBlockFrom(src, start);
}

function removeBlockFrom(src, start) {
  let i = src.indexOf("{", start);
  if (i === -1) return src;
  let depth = 0;
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") {
      depth--;
      if (depth === 0) {
        let end = i + 1;
        if (src[end] === ";") end++;
        return src.slice(0, start) + src.slice(end);
      }
    }
  }
  return src;
}

// Remove state block
src = src.replace(
  /\/\*\* @type \{GPUDevice \| null\} \*\/\nlet device = null;[\s\S]*?let builtEpoch = -1;\n\n/,
  "",
);

// Remove moved functions
for (const fn of [
  "packDrawParamsIso", "normalizeRgbStops", "packDrawParamsBeer", "packSsaoParams", "writeLayerColors",
  "ensureVolumeBuf", "stopsFromLayer", "compileChecked",
]) {
  src = removeFunction(src, fn);
}
for (const fn of [
  "uploadSceneColors", "uploadSceneVolumes", "setConstraintKeyframeBlends",
  "patchConstraintKeyframeFrame", "hasUploadedVolume", "ensurePipelinesForDegree",
]) {
  src = removeExportFunction(src, fn);
}

// Remove duplicate defaults and MAX_DENS at top if present
src = src.replace(
  /import \{ hexToRgb01, EXPR_GRADIENTS, MAX_GRAD_STOPS \} from[^\n]+\n\nexport const MAX_DENS_LAYERS = 8;\n\nconst DEFAULT_DENS_RGB[\s\S]*?let isoInterpHermite = true;\n\n/,
  "",
);

const header = `/**
 * WebGPU volume march: IDCT dens grids + iso manifolds + multi-layer Beer.
 */
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "../camera.js";
import { MAX_GRAD_STOPS } from "../../model/expressions.js";
import {
  gpu,
  MAX_DENS_LAYERS,
  PIPELINE_EPOCH,
  labelVertScratch,
  resetPipelinesOnDeviceLost,
  DEFAULT_DENS_RGB,
  DEFAULT_DENS_RGB2,
  DEFAULT_ISO_RGB,
  DEFAULT_ISO_RGB2,
} from "./gpuState.js";
import {
  packDrawParamsIso,
  packDrawParamsBeer,
  packSsaoParams,
  normalizeRgbStops,
  writeLayerColors,
} from "./uniforms.js";
import {
  uploadSceneColors,
  uploadSceneVolumes,
  setConstraintKeyframeBlends,
  patchConstraintKeyframeFrame,
  hasUploadedVolume,
  ensureVolumeBuf,
} from "./sceneUpload.js";
import { ensurePipelinesForDegree as buildPipelines } from "./pipelines.js";

export {
  MAX_DENS_LAYERS,
  uploadSceneColors,
  uploadSceneVolumes,
  setConstraintKeyframeBlends,
  patchConstraintKeyframeFrame,
  hasUploadedVolume,
};

`;

if (!src.startsWith("/**")) {
  throw new Error("unexpected march.js start");
}
src = header + src.replace(/^\/\*\*[\s\S]*?import \{ getIsoShader[\s\S]*?\n\n/, "");

// iso get/set use gpu
src = src.replace(
  /export function getIsoInterpHermite\(\) \{\s*return isoInterpHermite;\s*\}/,
  "export function getIsoInterpHermite() {\n  return gpu.isoInterpHermite;\n}",
);
src = src.replace(
  /export function setIsoInterpHermite\(on\) \{[\s\S]*?isoPipeline = null;\s*return true;\s*\}/,
  `export function setIsoInterpHermite(on) {
  const next = !!on;
  if (next === gpu.isoInterpHermite) return false;
  gpu.isoInterpHermite = next;
  gpu.isoPipeline = null;
  return true;
}`,
);

for (const v of stateVars) {
  const re = new RegExp(`(?<!gpu\\.)\\b${v}\\b`, "g");
  src = src.replace(re, `gpu.${v}`);
}

// Fix double gpu.gpu.
src = src.replace(/gpu\.gpu\./g, "gpu.");

// Fix writeLayerColors calls
src = src.replace(/writeLayerColors\(\[\[DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2\]\]\)/g,
  "writeLayerColors(gpu.device, gpu.colorBuf, [[DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2]])");
src = src.replace(/writeLayerColors\(gpu\.densGradStops\)/g,
  "writeLayerColors(gpu.device, gpu.colorBuf, gpu.densGradStops)");

// ensurePipelinesForDegree wrapper
src = src.replace(
  /await ensurePipelinesForDegree\(/g,
  "await ensurePipelinesForDegree(",
);
if (!src.includes("async function ensurePipelinesForDegree")) {
  src = src.replace(
    /export async function initClipBakeGpu/,
    `export async function ensurePipelinesForDegree(deg) {
  const result = await buildPipelines(deg);
  if (result && result.gridRebuildHalf != null) {
    syncClipGpuWorldGrid(result.gridRebuildHalf);
  }
  return result !== false;
}

export async function initClipBakeGpu`,
  );
}

// Fix device.lost callback - resetPipelinesOnDeviceLost
src = src.replace(
  /gpu\.device\.lost\.then\(\(\) => \{\s*gpu\.device = null;\s*gpu\.isoPipeline = gpu\.beerPipeline = gpu\.fxaaPipeline = gpu\.ssaoPipeline = gpu\.gridPipeline = gpu\.labelPipeline = null;\s*gpu\.labelAtlasTex = null;\s*gpu\.labelAtlasSamp = null;\s*gpu\.labelVertexBuf = null;\s*gpu\.labelAtlasDirty = true;\s*gpu\.initFailed = true;\s*\}\);/,
  `gpu.device.lost.then(() => {
        gpu.device = null;
        resetPipelinesOnDeviceLost();
        gpu.initFailed = true;
      });`,
);

// Remove duplicate PIPELINE_EPOCH const if any
src = src.replace(/\nconst PIPELINE_EPOCH = 25;\n/, "\n");

fs.writeFileSync(marchPath, src);
console.log("Refactored march.js, lines:", src.split("\n").length);
