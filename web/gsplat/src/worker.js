import { renderFrame } from "./rasterizer.js";

let scene = null;

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "scene") {
    scene = msg.scene;
    self.postMessage({ type: "ready", count: scene.count });
    return;
  }
  if (msg.type === "render") {
    if (!scene) return;
    const t0 = performance.now();
    const result = renderFrame(scene, msg.camera, msg.opts);
    // Transfer the buffer for speed
    const buffer = result.rgba.buffer;
    self.postMessage(
      {
        type: "frame",
        width: result.width,
        height: result.height,
        stats: result.stats,
        rgba: buffer,
        reqId: msg.reqId,
        queueMs: t0 - msg.sentAt,
      },
      [buffer],
    );
  }
};
