import { gpu } from "./gpuState.js";
import { noteGpuPresent, scheduleStampReadback } from "./marchProfile.js";

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
let stampEndWritten = false;
let stampedThisFrame = false;

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
  stampEndWritten = false;
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

function stampWrites(phase: "begin" | "end"): GPURenderPassTimestampWrites | undefined {
  if (!gpu.timestampsSupported || !gpu.stampQuerySet || gpu.stampReadPending) return undefined;
  if (phase === "begin") {
    if (stampBeginWritten) return undefined;
    stampBeginWritten = true;
    stampedThisFrame = true;
    return { querySet: gpu.stampQuerySet, beginningOfPassWriteIndex: 0 };
  }
  if (!stampBeginWritten || stampEndWritten) return undefined;
  stampEndWritten = true;
  return { querySet: gpu.stampQuerySet, endOfPassWriteIndex: 1 };
}

/** Attach timestamp query writes for the first or last pass of the frame. */
export function withStampWrites(
  desc: GPURenderPassDescriptor,
  phase: "begin" | "end",
): GPURenderPassDescriptor {
  const writes = stampWrites(phase);
  if (writes) desc.timestampWrites = writes;
  return desc;
}

function resolveStampQueries(device: GPUDevice): void {
  if (!stampedThisFrame || !stampEndWritten) return;
  if (!gpu.stampQuerySet || !gpu.stampResolveBuf || !gpu.stampReadBuf) return;
  if (gpu.stampReadPending) return;
  const enc = device.createCommandEncoder();
  enc.resolveQuerySet(gpu.stampQuerySet, 0, 2, gpu.stampResolveBuf, 0);
  enc.copyBufferToBuffer(gpu.stampResolveBuf, 0, gpu.stampReadBuf, 0, 16);
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
