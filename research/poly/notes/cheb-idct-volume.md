# Chebyshev IDCT volume bake (fiber-free)

**Golden path (target / shipping):** fit a tensor Chebyshev density on the box, **materialize dens on a Chebyshev node grid via separable IDCT**, then Beer–Lambert raymarch by sampling that volume. No per-view fiber compose / dens atlas.

Supersedes the clip-fiber / Babbage atlas path for the explorer’s preferred mode. Legacy fiber notes: [`clip-space-babbage.md`](./clip-space-babbage.md). Path C: [`path-c.md`](./path-c.md).

---

## Pipeline

```
Fit (CPU) → Chebyshev coeffs c_ijk on [-half,half]³
                │
                ▼  fit / coeff change only (view-independent)
     Separable 3D IDCT-III → dens at Chebyshev roots
       ξ_m = cos(π(m+½)/M),  M ≥ N+1
                │
                ▼  every frame
     Fullscreen march: box ray → sample volume → Beer F2B
```

| Stage | What | Cost |
|-------|------|------|
| Fit | Sample \(f\) @ Cheb nodes → DCT → \(c_{ijk}\) | once per expr |
| Bake | IDCT \(c \to\) dens grid \(M^3\) | \(O(M^3\log M)\) (≈ \(O(N^3\log N)\) if \(M\sim N\)) |
| March | Trilinear (in Cheb-index space) + Beer | \(O(\mathrm{steps})\) per pixel |

**Camera moves do not rebake.** Only coefficient / degree / half changes do.

---

## Why IDCT (not fiber compose)

Fiber bake restricts the 3D poly onto each screen ray (\(O(N^3)\)–\(O(N^4)\) per seed, \(\sim O(N^5)\) over a screen lattice of seeds). That couples bake to the view.

If \(f\) is already a Chebyshev tensor

\[
f(\xi,\eta,\zeta)=\sum_{i,j,k=0}^{N} c_{ijk}\,T_i(\xi)\,T_j(\eta)\,T_k(\zeta),
\qquad \xi=x/\mathrm{half},\;\ldots
\]

then values on the **tensor Chebyshev grid**

\[
\xi_m=\cos\Bigl(\pi\frac{m+\tfrac12}{M}\Bigr),\quad m=0,\ldots,M-1
\]

are exactly a **separable inverse DCT (IDCT-III)** along \(x\), then \(y\), then \(z\):

1. **Along \(x\):** for each of \(N{\times}N\) pairs \((j,k)\), length-\(N\!\to\!M\) IDCT → \(O(N^2 M\log M)\).
2. **Along \(y\):** for each of \(M{\times}N\) pairs \((m,k)\) → \(O(N M^2\log M)\).
3. **Along \(z\):** for each of \(M{\times}M\) pairs \((m,n)\) → \(O(M^3\log M)\).

With \(M\sim N\): **\(O(N^3\log N)\)** total — and the result lives in **box space**, not clip fibers.

Rays are diagonal in \((\xi,\eta,\zeta)\); the volume is axis-aligned. March **samples** the grid (lerp in Chebyshev-index coordinates), it does not store a 1D \(\gamma(u)\) per pixel.

---

## Discrete convention (match `fit.js`)

Fit uses Chebyshev roots of the first kind and the usual orthonormal-ish scaling (\(\alpha_0=1\), \(\alpha_{k>0}=2\)):

\[
c_{ijk}=\frac{\alpha_i\alpha_j\alpha_k}{n^3}\sum_{a,b,c} f(\xi_a,\xi_b,\xi_c)\,T_i(\xi_a)T_j(\xi_b)T_k(\xi_c).
\]

IDCT at the same nodes (\(M=n=N{+}1\)) recovers the interpolant samples. For \(M>N{+}1\), zero-pad \(c\) in each mode and IDCT on the finer root grid (same polynomial, denser samples → smoother trilinear).

Univariate prototype:

\[
v_m=\sum_{i=0}^{N} \tilde\alpha_i\, c_i\,\cos\Bigl(i\pi\frac{m+\tfrac12}{M}\Bigr)
\]

with \(\tilde\alpha\) matching the fit’s inverse (implementation: same factors as a known-good 1D round-trip against `fit.js` samples).

---

## March sampling

Box intersect in world space. At sample \(p\):

\[
\xi=\mathrm{clamp}(p/\mathrm{half},-1,1),\qquad
j_x=\frac{M}{\pi}\arccos(\xi_x)-\tfrac12
\]

(and likewise \(j_y,j_z\)). Trilinear dens in continuous index \((j_x,j_y,j_z)\). Outside the box slab → dens 0. Then Beer as usual.

Index-space lerp is exact at nodes and a smooth approximation between them; raising \(M\) (zero-padded IDCT) reduces lerp error without raising fit degree.

---

## Cost vs fiber atlas (N = 12 example)

| | Fiber γ / dens atlas | IDCT volume |
|--|----------------------|-------------|
| Bake trigger | every view / frame | fit / coeff change |
| Bake cost | ~14 ms seed @ 224×144 | \(\sim 13^3\) IDCT ≪ 1 ms CPU |
| March | atlas fetch + 1D eval | 3D fetch + lerp |
| Time-varying \(c(t)\) | full view bake | IDCT only |

---

## One-paragraph abstract

*Polynomial density is fit as a Chebyshev tensor on the axis-aligned box; a separable IDCT materializes density on a Chebyshev node lattice in \(O(N^3\log N)\). Realtime views never recompose fibers — they only raymarch and interpolate that static (or slowly updating) volume with Beer–Lambert integration.*
