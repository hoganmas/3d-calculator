import { gpu } from "./gpuState.js";
import { noteGpuPresent, scheduleStampReadback } from "./marchProfile.js";

/**
 * Query-set layout for the per-frame GPU timestamp breakdown:
 *   0 = begin (first pass of the frame)
 *   1 = end of iso/march stage
 *   2 = end of beer/volume stage
 *   3 = end of flow-particles stage
 *   4 = end of fxaa stage
 *   5 = end of grid-overlay stage (previously untimestamped entirely)
 * Stage duration = consecutive delta (e.g. beer stage = t[2] - t[1]).
 */
export const STAMP_COUNT = 6;
export const STAMP_END_MARCH = 1;
export const STAMP_END_BEER = 2;
export const STAMP_END_FLOW = 3;
export const STAMP_END_FXAA = 4;
export const STAMP_END_GRID = 5;

/** Command buffers collected between `beginGpuFrame` / `endGpuFrame`. */
let frameCmds: GPUCommandBuffer[] | null = null;
/**
 * Every buffer `gpuWriteBuffer` touched since the last queue submit.
 * Flushing only the *last* write misses: coarse iso → upsample uniforms →
 * refine iso, which reused `drawParamBuf` with display-sized fbW/fbH while
 * the coarse pass was still pending. That parks NDC (0,0) at the bottom-right
 * of the smaller coarse target.
 */
const writtenSinceSubmit = new Set<GPUBuffer>();
let stampBeginWritten = false;
/** Bitmask of which end-of-stage indices (1..STAMP_COUNT-1) have been written this frame. */
let stampEndWrittenMask = 0;
let stampedThisFrame = false;
/** All end-of-stage indices (1..STAMP_COUNT-1) written — i.e. every stage boundary landed. */
const STAMP_END_FULL_MASK = (1 << STAMP_COUNT) - 2;

function flushPendingCmds(device: GPUDevice): void {
  if (!frameCmds || frameCmds.length === 0) return;
  device.queue.submit(frameCmds);
  frameCmds.length = 0;
  writtenSinceSubmit.clear();
}

/** Rewrite of a buffer already used by pending cmds must submit those cmds first. */
export function uniformWriteNeedsFlush<T>(
  pendingCmdCount: number,
  writtenSinceSubmit: ReadonlySet<T>,
  buf: T,
): boolean {
  return pendingCmdCount > 0 && writtenSinceSubmit.has(buf);
}

export function beginGpuFrame(): void {
  frameCmds = [];
  writtenSinceSubmit.clear();
  stampBeginWritten = false;
  stampEndWrittenMask = 0;
  stampedThisFrame = false;
}

/**
 * Queue a buffer write. If pending command buffers may still read this
 * buffer, submit them first so they keep the earlier contents.
 */
export function gpuWriteBuffer(
  device: GPUDevice,
  buf: GPUBuffer,
  data: ArrayBufferView | ArrayBuffer,
  offset = 0,
): void {
  if (frameCmds && frameCmds.length > 0 && uniformWriteNeedsFlush(frameCmds.length, writtenSinceSubmit, buf)) {
    flushPendingCmds(device);
  }
  device.queue.writeBuffer(buf, offset, data);
  writtenSinceSubmit.add(buf);
}

export function submitEnc(device: GPUDevice, enc: GPUCommandEncoder): void {
  const cmd = enc.finish();
  if (frameCmds) frameCmds.push(cmd);
  else device.queue.submit([cmd]);
}

function stampBeginWrites(): GPURenderPassTimestampWrites | undefined {
  if (!gpu.timestampsSupported || !gpu.stampQuerySet || gpu.stampReadPending) return undefined;
  if (stampBeginWritten) return undefined;
  stampBeginWritten = true;
  stampedThisFrame = true;
  return { querySet: gpu.stampQuerySet, beginningOfPassWriteIndex: 0 };
}

function stampEndWrites(index: number): GPURenderPassTimestampWrites | undefined {
  if (!gpu.timestampsSupported || !gpu.stampQuerySet || gpu.stampReadPending) return undefined;
  if (stampEndWrittenMask & (1 << index)) return undefined;
  stampEndWrittenMask |= 1 << index;
  return { querySet: gpu.stampQuerySet, endOfPassWriteIndex: index };
}

/**
 * Attach a timestamp write to a render pass descriptor: `"begin"` for the
 * frame's first pass, or one of the `STAMP_END_*` indices for a stage
 * boundary that already has a real pass to attach to.
 */
export function withStampWrites(
  desc: GPURenderPassDescriptor,
  phase: "begin" | number,
): GPURenderPassDescriptor {
  const writes = phase === "begin" ? stampBeginWrites() : stampEndWrites(phase);
  if (writes) desc.timestampWrites = writes;
  return desc;
}

/**
 * Guarantee a stage-boundary timestamp gets written even when that stage did
 * no real GPU work this frame (no isosurfaces, no flow, grid off, etc) — a
 * trivial no-op pass (load/store, zero draws) on an existing texture, so the
 * breakdown always has every STAMP_COUNT slot filled and stage deltas read
 * as ~0 for a skipped stage instead of misaligning the ones after it.
 */
export function stampCheckpoint(device: GPUDevice, view: GPUTextureView, index: number): void {
  const writes = stampEndWrites(index);
  if (!writes) return;
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view, loadOp: "load", storeOp: "store" }],
    timestampWrites: writes,
  });
  pass.end();
  submitEnc(device, enc);
}

function resolveStampQueries(device: GPUDevice): void {
  if (!stampedThisFrame || !stampBeginWritten || stampEndWrittenMask !== STAMP_END_FULL_MASK) return;
  if (!gpu.stampQuerySet || !gpu.stampResolveBuf || !gpu.stampReadBuf) return;
  if (gpu.stampReadPending) return;
  const enc = device.createCommandEncoder();
  enc.resolveQuerySet(gpu.stampQuerySet, 0, STAMP_COUNT, gpu.stampResolveBuf, 0);
  enc.copyBufferToBuffer(gpu.stampResolveBuf, 0, gpu.stampReadBuf, 0, STAMP_COUNT * 8);
  submitEnc(device, enc);
}

export function endGpuFrame(device: GPUDevice): void {
  resolveStampQueries(device);
  if (frameCmds && frameCmds.length > 0) device.queue.submit(frameCmds);
  frameCmds = null;
  writtenSinceSubmit.clear();
}

/** Record present cadence; GPU time comes from timestamp readback. */
export function sampleGpuPresent(submitWallAt: number): void {
  noteGpuPresent(submitWallAt);
  scheduleStampReadback();
}
