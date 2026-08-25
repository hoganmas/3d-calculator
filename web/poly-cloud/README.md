# Polynomial density cloud viewer (Three.js)

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial, convert to a **world monomial tensor** \(c_{ijk}\).
2. Custom shader: each pixel **contracts** that tensor with the camera ray \(p(t)=o+td\) into a 1D poly \(\alpha(t)\).
3. Raymarch / Path C uses only **Horner** (and analytic \(\int\alpha\)) along the ray.

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

## Cost

| Stage | Cost |
|---|---|
| Fit (CPU, once) | Chebyshev + monomial convert |
| Per pixel | \(O((N+1)^3)\) build \(\alpha\) |
| Per sample | \(O(N)\) Horner |

Previously every sample paid \((N+1)^3\) for a full 3D Chebyshev eval.

## Controls

- **steps** — samples along the ray
- **Path C** — \(\Delta\tau=\int\alpha\,dt\) per step from the 1D poly
