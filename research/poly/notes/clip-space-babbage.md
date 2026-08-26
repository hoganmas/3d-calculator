# Clip-grid dens bake (legacy fiber / Babbage atlas)

**Superseded** for the explorer golden path by [`cheb-idct-volume.md`](./cheb-idct-volume.md) (Chebyshev IDCT box volume + sample march).

This note documents the earlier **view-dependent fiber atlas** approach (compose / dens samples along screen rays, Chebyshev+Clenshaw fill, Beer march from the atlas). Kept for research context.

## Legacy pipeline (fibers)

```
Fit → world monomials
         ▼ camera / LOD
GPU dens atlas (Cheb seeds + Clenshaw in x)
         ▼
Fullscreen march (bilinear + u-eval → Beer)
```

Fiber restriction couples bake to the view and cost scales poorly with seed count (\(\sim O(N^5)\) framing). The IDCT volume path moves dens into box space so orbit does not rebake.

See also [`path-c.md`](./path-c.md).
