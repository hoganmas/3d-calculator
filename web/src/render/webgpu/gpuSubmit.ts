import { gpu } from "./gpuState.js";
import { noteGpuPresent, scheduleStampReadback } from "./marchProfile.js";

/** Command buffers collected between `beginGpuFrame` / `endGpuFrame`. */
let frameCmds: GPUCommandBuffer[] | null = null;
/** Last uniform buffer written while cmds are pending (flush if rewritten). */
let lastWrittenBuf: GPUBuffer | null = null;
let stampBeginWritten = false;
let stampEndWritten = false;
let stampedThisFrame = false;

export function beginGpuFrame(): void {
  frameCmds = [];
  lastWrittenBuf = null;
  stampBeginWritten = false;
  stampEndWritten = false;
  stampedThisFrame = false;
}

/**
 * Queue a buffer write. If that same buffer is already referenced by a pending
 * command buffer, submit first so the GPU sees the earlier contents.
 */
export function gpuWriteBuffer(
  device: GPUDevice,
  buf: GPUBuffer,
  data: ArrayBufferView | ArrayBuffer,
  offset = 0,
): void {
  if (frameCmds && frameCmds.length > 0 && lastWrittenBuf === buf) {
    device.queue.submit(frameCmds);
    frameCmds.length = 0;
  }
  device.queue.writeBuffer(buf, offset, data);
  lastWrittenBuf = buf;
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
  lastWrittenBuf = null;
}

/** Record present cadence; GPU time comes from timestamp readback. */
export function sampleGpuPresent(submitWallAt: number): void {
  noteGpuPresent(submitWallAt);
  scheduleStampReadback();
}
