# WebXR (immersive view)

Headset-agnostic **immersive-vr** viewing via Three.js `WebXRManager`. Expression editing, settings, and share stay on the flat DOM UI for now.

## What works in XR

- Stereo **WebGL Beer** volume march (same path as the desktop WebGPU fallback)
- Look via headset pose (`local` / `local-floor` when available)
- **Squeeze** on a controller: grab-move/rotate the volume (`xrWorld`)
- **Trigger / select**: recenter the volume in front of you
- Toolbar **Focus view** / MCP `laplacian_reset_camera`: same recenter while presenting
- Toolbar **Enter XR** / **Exit XR** (button is hidden when `immersive-vr` is unsupported)

## What does not run in XR (yet)

- WebGPU multi-layer march (iso SSAO/FXAA, IBFV, particles) — forced off while presenting
- MathLive sidebar, param rail, settings dialogs
- Axis letter overlay canvas (grid/axes lines still show on the WebGL path)

## Requirements

- Secure context: **HTTPS** or **localhost**
- A browser with WebXR `immersive-vr` (Quest Browser, Chrome + headset, etc.)

## Local smoke checklist

1. `cd web && npm install && npm run dev`
2. Open the app on **localhost** (or deploy HTTPS).
3. **Chrome WebXR emulator** (Immersive Web Emulator / WebXR API Emulator):
   - Confirm **Enter XR** appears in the viewport toolbar.
   - Enter session: volume should appear ahead; WebGPU overlay hidden.
   - Squeeze-drag moves the volume; trigger recenters; Exit XR restores OrbitControls.
4. **Real headset** (optional): same flow over HTTPS; prefer `local-floor` standing/seated.

## Architecture notes

| Piece | Location |
|-------|----------|
| Session + Enter XR | `src/app/xr/session.ts` |
| Grab / recenter | `src/app/xr/nav.ts` |
| `xrWorld` content root | `src/app/scene.ts` |
| Force WebGL + per-eye uniforms | `src/app/webglFallback.ts` (`useGpuClipPath`, `clipQuad.onBeforeRender`) |
| XR-aware rAF | `src/app/loop.ts` (`renderer.setAnimationLoop`) |

Later: DOM/XR UI port, then optional `XRGPUBinding` for WebGPU stereo when portable enough.
