# Laplaci

Multi-expression 3D polynomial calculator (Three.js + WebGPU).

**Live demo:** [laplaci.com](https://laplaci.com/)

**Pipeline:** Fit each expression with a Chebyshev polynomial → IDCT dens volume → draw manifolds (opaque isosurfaces) then density clouds (Beer–Lambert), with shared free parameters.

Details: [`research/poly/notes/cheb-idct-volume.md`](../research/poly/notes/cheb-idct-volume.md).

Source layout: [`STRUCTURE.md`](STRUCTURE.md). WebXR immersive view notes: [`docs/webxr.md`](docs/webxr.md).

## Run locally

```bash
npm install
npm run dev
```

WebXR requires a secure context (`localhost` or HTTPS). See [`docs/webxr.md`](docs/webxr.md) for the Enter XR smoke checklist.
## Tests

Simulation correctness (fit → IDCT → volume) runs in Node — no browser needed:

```bash
npm test              # all suites
npm test flow         # flow parse + fit
npm test cloud        # cloud density
npm test isosurface   # isosurface gradients
```

See [`tests/README.md`](tests/README.md) for layout. Integration smoke tests (`test:integration`) need `npm run dev` running.

## Build

Production build uses `base: /` for deployment at [laplaci.com](https://laplaci.com/):

```bash
npm run build
npm run preview
```

GitHub Actions (`.github/workflows/pages.yml`) deploys `dist/` on push to `main`. Enable **Settings → Pages → Build and deployment → GitHub Actions** once if needed.

Requires a WebGPU-capable browser for the full multi-layer / manifold path. Without WebGPU, a single summed density volume still marches in WebGL.

## MCP setup

Click **Setup MCP** in the header to copy server config for the current page. Requires Node.js (`npx`), an MCP client, and this tab open in a system browser (Chrome, Firefox, or Safari).

WebMCP is **off by default** (avoids Chrome’s “access other apps and services” permission on casual visits). Enable via **Setup MCP** in the header, `?webmcp=1` on the URL, or `localStorage.setItem("laplacian-webmcp", "1")`. Disable with `?webmcp=0`. Agents should start with `laplacian_get_capabilities`.

Optional dev CLI: `node scripts/webmcp-relay-call.mjs laplacian_setup_lava_lamp` (uses an existing tab only).

## Controls

- **Expressions** — Enter splits / adds a row. Free symbols get a slider on that same row; write `a=…` for an explicit parameter equation. `A=B` → isosurface; bare / `f=…` → cloud; `(Fx,Fy,Fz)` or `\grad f` → flow.
- **Focus view** (crosshair icon, header) — reset camera to the default view.
- **Setup MCP** (chain icon, header) — copy server config for the current page.
- **Settings** (gear) — appearance (theme), preset, poly deg / scale / steps / box size, march downscale, diagnostics.

## Cost

| Stage | When |
|---|---|
| Fit + IDCT | Expression / deg / box / param change |
| March | Every frame (camera / hyperparams) |
