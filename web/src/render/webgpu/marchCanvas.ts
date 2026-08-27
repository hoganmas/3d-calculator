import { gpu } from "./gpuState.js";
import type { CanvasSize } from "./marchTypes.js";

function destroyTexture(tex: GPUTexture | null): void {
  if (!tex) return;
  try { tex.destroy(); } catch { /* device lost */ }
}

function ensureOcclIsoTex(w: number, h: number): void {
  if (gpu.occlIsoTex && gpu.occlIsoW === w && gpu.occlIsoH === h) return;
  destroyTexture(gpu.occlIsoTex);
  if (!gpu.device) return;
  gpu.occlIsoTex = gpu.device.createTexture({
    size: [w, h],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.occlIsoW = w;
  gpu.occlIsoH = h;
}

function ensureOcclSurfTex(w: number, h: number): void {
  if (gpu.occlSurfTex && gpu.occlSurfW === w && gpu.occlSurfH === h) return;
  destroyTexture(gpu.occlSurfTex);
  if (!gpu.device) return;
  gpu.occlSurfTex = gpu.device.createTexture({
    size: [w, h],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.occlSurfW = w;
  gpu.occlSurfH = h;
}

function ensureDepthTex(w: number, h: number): void {
  if (gpu.depthTex && gpu.depthW === w && gpu.depthH === h) return;
  destroyTexture(gpu.depthTex);
  if (!gpu.device) return;
  gpu.depthTex = gpu.device.createTexture({
    size: [w, h],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  gpu.depthW = w;
  gpu.depthH = h;
}

function ensureNormalTex(w: number, h: number): void {
  if (gpu.normalTex && gpu.normalW === w && gpu.normalH === h) return;
  destroyTexture(gpu.normalTex);
  if (!gpu.device) return;
  gpu.normalTex = gpu.device.createTexture({
    size: [w, h],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.normalW = w;
  gpu.normalH = h;
}

function ensureSceneColorTex(w: number, h: number): void {
  if (gpu.sceneColorTex && gpu.sceneColorW === w && gpu.sceneColorH === h) return;
  destroyTexture(gpu.sceneColorTex);
  if (!gpu.device) return;
  gpu.sceneColorTex = gpu.device.createTexture({
    size: [w, h],
    format: gpu.canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.sceneColorW = w;
  gpu.sceneColorH = h;
}

function ensureSceneColorAoTex(w: number, h: number): void {
  if (gpu.sceneColorAoTex && gpu.sceneColorAoW === w && gpu.sceneColorAoH === h) return;
  destroyTexture(gpu.sceneColorAoTex);
  if (!gpu.device) return;
  gpu.sceneColorAoTex = gpu.device.createTexture({
    size: [w, h],
    format: gpu.canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.sceneColorAoW = w;
  gpu.sceneColorAoH = h;
}

export function ensureMarchTargets(w: number, h: number): void {
  ensureOcclIsoTex(w, h);
  ensureOcclSurfTex(w, h);
  ensureDepthTex(w, h);
  ensureNormalTex(w, h);
  ensureSceneColorTex(w, h);
  ensureSceneColorAoTex(w, h);
}

function attachClipGpuCanvas(viewportEl: HTMLElement): HTMLCanvasElement {
  if (gpu.canvas) return gpu.canvas;
  gpu.canvas = document.createElement("canvas");
  gpu.canvas.className = "clip-gpu";
  gpu.canvas.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;display:none;";
  viewportEl.appendChild(gpu.canvas);
  return gpu.canvas;
}

export function setClipGpuCanvasVisible(visible: boolean): void {
  if (!gpu.canvas) return;
  gpu.canvas.style.display = visible ? "block" : "none";
}

export function resizeClipGpuCanvas(pixelW: number, pixelH: number): CanvasSize {
  if (!gpu.canvas || !gpu.device) return { w: pixelW | 0, h: pixelH | 0 };
  if (!gpu.ctx) {
    gpu.ctx = gpu.canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!gpu.ctx) return { w: pixelW | 0, h: pixelH | 0 };
  }
  const w = Math.max(1, pixelW | 0);
  const h = Math.max(1, pixelH | 0);
  const changed = gpu.canvas.width !== w || gpu.canvas.height !== h;
  if (changed) {
    gpu.canvas.width = w;
    gpu.canvas.height = h;
  }
  gpu.canvas.style.width = "100%";
  gpu.canvas.style.height = "100%";
  if (changed || !gpu.canvas._clipConfigured) {
    gpu.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    gpu.ctx.configure({
      device: gpu.device,
      format: gpu.canvasFormat,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    gpu.canvas._clipConfigured = true;
  }
  return { w: gpu.canvas.width, h: gpu.canvas.height };
}

/** Clear the WebGPU overlay to transparent (no density / iso). */
export function clearClipGpuFrame(fbW: number, fbH: number): boolean {
  if (!gpu.device || !gpu.ctx || !gpu.canvas) return false;
  const { w, h } = resizeClipGpuCanvas(fbW, fbH);
  const view = gpu.ctx.getCurrentTexture().createView();
  const enc = gpu.device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.end();
  gpu.device.queue.submit([enc.finish()]);
  gpu.profileMarchFbW = w;
  gpu.profileMarchFbH = h;
  gpu.profileMethod = "gpu-clear";
  return true;
}

export function attachMarchCanvas(viewportEl: HTMLElement): HTMLCanvasElement {
  return attachClipGpuCanvas(viewportEl);
}

export function bindMarchCanvasContext(): void {
  if (gpu.canvas) {
    gpu.ctx = gpu.canvas.getContext("webgpu") as GPUCanvasContext | null;
  }
}
