import type { PerspectiveCamera } from "three";
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "../camera.js";
import {
  gpu,
  DEFAULT_ISO_RGB,
  DEFAULT_ISO_RGB2,
} from "./gpuState.js";
import {
  packDrawParamsIso,
  packDrawParamsBeer,
  packSsaoParams,
  writeLayerColors,
} from "./uniforms.js";
import { packGridParams, syncClipGpuWorldGrid, uploadAxisLabelBillboards } from "./gridOverlay.js";
import { ensureMarchTargets, resizeClipGpuCanvas } from "./marchCanvas.js";
import { noteGpuPresent, scheduleStampReadback } from "./marchProfile.js";
import {
  acquireMarchGpuHandles,
  acquireMarchTargets,
  type MarchGpuHandles,
  type MarchTargets,
  type RenderClipFrameGpuParams,
} from "./marchTypes.js";

function buildRaySetup(
  params: RenderClipFrameGpuParams,
): {
  handles: MarchGpuHandles;
  targets: MarchTargets;
  ro: [number, number, number];
  dirMatrix: ReturnType<typeof offsetDirMatrix>;
  half: number;
  marchW: number;
  marchH: number;
  outW: number;
  outH: number;
} | null {
  if (gpu.densLayerCount < 1 && gpu.sceneConstraints.length < 1) return null;

  const handles = acquireMarchGpuHandles();
  if (!handles) return null;

  const { camera, half, fbW, fbH, ndcOffsetX = 0, ndcOffsetY = 0, displayW = 0, displayH = 0 } = params;
  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const dirMatrix = offsetDirMatrix(ndcToDirMatrix(camera, sx, sy), ndcOffsetX, ndcOffsetY);
  const ro: [number, number, number] = [o.x, o.y, o.z];
  const h = half ?? 2;

  const marchW = Math.max(1, fbW | 0);
  const marchH = Math.max(1, fbH | 0);
  const outW = Math.max(1, (displayW | 0) || marchW);
  const outH = Math.max(1, (displayH | 0) || marchH);

  resizeClipGpuCanvas(outW, outH);
  ensureMarchTargets(marchW, marchH);
  syncClipGpuWorldGrid(h);

  const targets = acquireMarchTargets();
  if (!targets) return null;

  return { handles, targets, ro, dirMatrix, half: h, marchW, marchH, outW, outH };
}

function clearMarchTargets(
  device: GPUDevice,
  targets: MarchTargets,
): void {
  const sceneView = targets.sceneColorTex.createView();
  const occlIsoView = targets.occlIsoTex.createView();
  const depthView = targets.depthTex.createView();
  const normalView = targets.normalTex.createView();

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: sceneView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
      {
        view: occlIsoView,
        clearValue: { r: 1, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
      {
        view: normalView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  pass.end();
  device.queue.submit([enc.finish()]);
}

function drawIsoConstraints(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  sceneView: GPUTextureView,
  marchW: number,
  marchH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): void {
  const { device, isoPipeline, drawParamBuf, volumeBuf } = handles;
  const occlIsoView = targets.occlIsoTex.createView();
  const depthView = targets.depthTex.createView();
  const normalView = targets.normalTex.createView();

  for (const c of gpu.sceneConstraints) {
    const stride = c.frameStride || 0;
    const base0 = c.base + (c.i0 | 0) * stride;
    const base1 = c.base + (c.i1 | 0) * stride;
    const blendT = Number.isFinite(c.t) ? c.t : 0;
    const c0 = c.color || DEFAULT_ISO_RGB;
    const c1 = c.color2 || DEFAULT_ISO_RGB2;
    const stops = c.colors || [c0, c1];
    device.queue.writeBuffer(
      drawParamBuf,
      0,
      packDrawParamsIso(
        marchW, marchH, Mgrid, steps, half, scale, c.isoLevel, base0, ro, dirMatrix,
        c0, c1,
        base1, blendT, stops,
      ),
    );
    const bg = device.createBindGroup({
      layout: isoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: drawParamBuf } },
        { binding: 1, resource: { buffer: volumeBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: sceneView, loadOp: "load", storeOp: "store" },
        { view: occlIsoView, loadOp: "load", storeOp: "store" },
        { view: normalView, loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(isoPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
}

function drawSsaoPass(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  sceneView: GPUTextureView,
  marchW: number,
  marchH: number,
  half: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): GPUTextureView {
  const { device, ssaoPipeline, ssaoParamBuf } = handles;
  const aoView = targets.sceneColorAoTex.createView();
  const occlIsoView = targets.occlIsoTex.createView();
  const normalView = targets.normalTex.createView();

  device.queue.writeBuffer(
    ssaoParamBuf,
    0,
    packSsaoParams(
      marchW, marchH, half,
      Math.max(0.2, half * 0.18),
      0.85,
      0.03,
      ro, dirMatrix,
    ),
  );
  const bg = device.createBindGroup({
    layout: ssaoPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ssaoParamBuf } },
      { binding: 1, resource: sceneView },
      { binding: 2, resource: occlIsoView },
      { binding: 3, resource: normalView },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: aoView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(ssaoPipeline);
  pass.setBindGroup(0, bg);
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
  return aoView;
}

function drawBeerPass(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  sceneView: GPUTextureView,
  marchW: number,
  marchH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): void {
  const { device, beerPipeline, drawParamBufBeer, volumeBuf, colorBuf } = handles;
  const occlIsoView = targets.occlIsoTex.createView();
  const occlSurfView = targets.occlSurfTex.createView();

  device.queue.writeBuffer(
    drawParamBufBeer,
    0,
    packDrawParamsBeer(marchW, marchH, Mgrid, steps, half, scale, gpu.densBase, gpu.densLayerCount, ro, dirMatrix),
  );
  const bg = device.createBindGroup({
    layout: beerPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: drawParamBufBeer } },
      { binding: 1, resource: { buffer: volumeBuf } },
      { binding: 2, resource: occlIsoView },
      { binding: 3, resource: { buffer: colorBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      { view: sceneView, loadOp: "load", storeOp: "store" },
      { view: occlSurfView, loadOp: "clear", clearValue: { r: 1, g: 0, b: 0, a: 1 }, storeOp: "store" },
    ],
  });
  pass.setPipeline(beerPipeline);
  pass.setBindGroup(0, bg);
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
}

function drawFxaaPass(
  handles: MarchGpuHandles,
  sceneView: GPUTextureView,
  swapView: GPUTextureView,
  marchW: number,
  marchH: number,
): void {
  const { device, fxaaPipeline, fxaaParamBuf, fxaaSampler } = handles;
  const inv = new Float32Array([1 / marchW, 1 / marchH, 0, 0]);
  device.queue.writeBuffer(fxaaParamBuf, 0, inv);
  const bg = device.createBindGroup({
    layout: fxaaPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: fxaaParamBuf } },
      { binding: 1, resource: sceneView },
      { binding: 2, resource: fxaaSampler },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: swapView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(fxaaPipeline);
  pass.setBindGroup(0, bg);
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
}

function drawGridOverlay(
  handles: MarchGpuHandles,
  occlForGrid: GPUTextureView,
  camera: PerspectiveCamera,
  swapView: GPUTextureView,
  half: number,
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
  ro: [number, number, number],
  outW: number,
  outH: number,
): void {
  if (!gpu.gridVertexBuf || gpu.gridVertexCount <= 0) return;

  const { device, gridPipeline, gridParamBuf } = handles;

  camera.updateMatrixWorld(true);
  const viewProj = new Float32Array(16);
  const e = camera.projectionMatrix.elements;
  const v = camera.matrixWorldInverse.elements;
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      viewProj[c * 4 + r] =
        e[0 * 4 + r] * v[c * 4 + 0] +
        e[1 * 4 + r] * v[c * 4 + 1] +
        e[2 * 4 + r] * v[c * 4 + 2] +
        e[3 * 4 + r] * v[c * 4 + 3];
    }
  }
  device.queue.writeBuffer(
    gridParamBuf,
    0,
    packGridParams(viewProj, ro, half, dirMatrix, outW, outH),
  );
  const labelVertCount = gpu.labelPipeline
    ? uploadAxisLabelBillboards(camera, half)
    : 0;
  const bg = device.createBindGroup({
    layout: gridPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gridParamBuf } },
      { binding: 1, resource: occlForGrid },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: swapView, loadOp: "load", storeOp: "store" }],
  });
  pass.setPipeline(gridPipeline);
  pass.setBindGroup(0, bg);
  pass.setVertexBuffer(0, gpu.gridVertexBuf);
  pass.draw(gpu.gridVertexCount);

  if (gpu.labelPipeline && gpu.labelVertexBuf && gpu.labelAtlasTex && gpu.labelAtlasSamp && labelVertCount > 0) {
    const labelPipeline = gpu.labelPipeline;
    const labelBg = device.createBindGroup({
      layout: labelPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gridParamBuf } },
        { binding: 1, resource: gpu.labelAtlasTex.createView() },
        { binding: 2, resource: gpu.labelAtlasSamp },
        { binding: 3, resource: occlForGrid },
      ],
    });
    pass.setPipeline(labelPipeline);
    pass.setBindGroup(0, labelBg);
    pass.setVertexBuffer(0, gpu.labelVertexBuf);
    pass.draw(labelVertCount);
  }

  pass.end();
  device.queue.submit([enc.finish()]);
}

export function renderClipFrameGpu(params: RenderClipFrameGpuParams): boolean {
  const setup = buildRaySetup(params);
  if (!setup) return false;

  const {
    handles, targets, ro, dirMatrix, half, marchW, marchH, outW, outH,
  } = setup;
  const { device, ctx, volumeBuf, colorBuf } = handles;
  const { camera, scale, steps } = params;
  const Mgrid = gpu.sceneM;

  gpu.profileMarchFbW = marchW;
  gpu.profileMarchFbH = marchH;
  gpu.profileMethod = "gpu-iso+ssao+beer+grid+fxaa";
  gpu.profileGridM = Mgrid;

  if (gpu.scenePacked) device.queue.writeBuffer(volumeBuf, 0, gpu.scenePacked);
  writeLayerColors(device, colorBuf, gpu.densGradStops);

  let sceneView = targets.sceneColorTex.createView();
  const swapView = ctx.getCurrentTexture().createView();

  clearMarchTargets(device, targets);
  drawIsoConstraints(handles, targets, sceneView, marchW, marchH, Mgrid, steps, half, scale, ro, dirMatrix);

  if (gpu.sceneConstraints.length > 0) {
    sceneView = drawSsaoPass(handles, targets, sceneView, marchW, marchH, half, ro, dirMatrix);
  }

  const ranBeer = gpu.densLayerCount > 0 && gpu.densPacked;
  if (ranBeer) {
    drawBeerPass(handles, targets, sceneView, marchW, marchH, Mgrid, steps, half, scale, ro, dirMatrix);
  }

  drawFxaaPass(handles, sceneView, swapView, marchW, marchH);
  const occlForGrid = ranBeer
    ? targets.occlSurfTex.createView()
    : targets.occlIsoTex.createView();
  drawGridOverlay(handles, occlForGrid, camera, swapView, half, dirMatrix, ro, outW, outH);

  const submitWallAt = performance.now();
  void device.queue.onSubmittedWorkDone().then(() => {
    noteGpuPresent(submitWallAt);
    scheduleStampReadback();
  });
  return true;
}
