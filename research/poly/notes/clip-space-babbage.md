# Clip-space homogenization + grid differences for polynomial volumes

Notes from the `web/poly-cloud` explorer discussion: how to avoid **per-ray \(\Theta(N^3)\)** contraction of a dense tensor-product field, using homogeneous coordinates, a per-view pullback to clip space, and Babbage-style forward differences on the uniform pixel grid.

Related: current viewer still does **per-pixel LOS composition** in world/eye parameter \(t\) (exact, \(\Theta(N^3)\) per ray). This document describes a **target architecture**, not the shipped shader.

---

## 1. Problem with the naive pipeline

Density is a dense tensor-product polynomial of degree \(N\) in each variable:

\[
f(x,y,z)=\sum_{i,j,k=0}^{N} c_{ijk}\,x^i y^j z^k
\qquad\big((N+1)^3\text{ coeffs}\big).
\]

For each camera ray \(\mathbf{p}(t)=\mathbf{o}+t\hat{\mathbf{d}}\) (or \(\mathbf{p}(t)=t\hat{\mathbf{d}}\) in camera frame), the restriction is a **univariate** polynomial \(\gamma(t)\) of degree \(\le 3N\). Forming \(\gamma\) by touching all coeffs costs **\(\Omega(N^3)\) per ray**. With \(W\times H\) pixels that dominate frame time.

Horner / Clenshaw along the ray is only \(\Theta(\mathrm{steps}\cdot 3N)\) *after* \(\gamma\) exists. Forward differences (Babbage) likewise only accelerate **1D evaluation** of an already-built univariate poly—not the 3D→1D extraction.

---

## 2. Perspective, NDC, and why “parameterize by \(z_\mathrm{ndc}\)” looks rational

Under perspective, a primary ray has **constant** \((x_\mathrm{ndc},y_\mathrm{ndc})\); only \(z_\mathrm{ndc}\) changes. All primary rays are parallel to \(+\hat{\mathbf{z}}\) in NDC.

Eye depth and NDC depth are related by a **Möbius** map:

\[
z_\mathrm{ndc}=\frac{a\,z_\mathrm{eye}+b}{c\,z_\mathrm{eye}+d}
\quad\Rightarrow\quad
z_\mathrm{eye}=\frac{\alpha z_\mathrm{ndc}+\beta}{\gamma z_\mathrm{ndc}+\delta}.
\]

Along a ray, \(x_\mathrm{eye},y_\mathrm{eye}\) scale with \(z_\mathrm{eye}\). So as functions of \(z_\mathrm{ndc}\), eye/world coordinates are **rational**. Then \(f(x_\mathrm{eye},y_\mathrm{eye},z_\mathrm{eye})\) is a **rational function of \(z_\mathrm{ndc}\)**, not a polynomial—unless you never divide.

Parameterizing by Euclidean ray \(t\) keeps \(f(t\hat{\mathbf{d}})\) polynomial, but each ray still needs its own \(\gamma\) at \(\Omega(N^3)\).

---

## 3. Homogeneous coordinates

Homogenize a degree-\(n\) Cartesian polynomial (for tensor degree \(N\), take \(n\le 3N\) or homogenize in a fixed total-degree embedding):

\[
F(X,Y,Z,W)=W^n\,f\!\left(\frac{X}{W},\frac{Y}{W},\frac{Z}{W}\right).
\]

\(F\) is a **homogeneous** polynomial. Projection \(\mathrm{clip}=P\cdot\mathrm{eye}_{\mathrm{homog}}\) is **linear** in homogeneous coordinates (no divide).

**Restriction of \(F\) to a projective line** (linear in a homogeneous line parameter \(\lambda\)) is again a **univariate polynomial** in \(\lambda\). Algebraically there is no Möbius.

The Cartesian density is recovered by dehomogenization:

\[
f=\frac{F}{W^n}
\]

(or the analogous factor in eye \(w\) after \(P^{-1}\)). So:

| Object | Along a ray (homogeneous param) |
|--------|----------------------------------|
| \(F\) (homogenized) | polynomial |
| \(f=F/W^n\) (physical density) | rational in that param unless \(W^n\) is handled in the measure |

Homogeneous language makes perspective linear **before** the divide; it does not by itself remove per-ray \(\Omega(N^3)\) work.

---

## 4. Target pipeline

### Step A — Homogenize (asset / fit time, or once when \(c_{ijk}\) change)

Build \(F\) from world/eye monomials (or keep an equivalent homogeneous tensor).

### Step B — Pull back to clip space (**once per view**, when \(P\) / camera changes)

\[
G(C)=F(P^{-1}C).
\]

\(G\) is still homogeneous in clip \((X,Y,Z,W)\). This is a **shared per-frame** linear change of variables on coefficients—not per pixel.

### Step C — Fibers on the uniform pixel grid

In an affine clip chart (e.g. \(W=1\), or work with \(\beta_k\) from \(G(xW,yW,Z,W)\)):

\[
G(xW,yW,Z,W)=\sum_{k=0}^{n}\beta_k(x,y)\,Z^k W^{n-k}.
\]

Each \(\beta_k(x,y)\) is a **bivariate** polynomial. Pixel centers \((x_i,y_j)\) form a **uniform grid** in clip/NDC \(x,y\).

**2D forward differences (Babbage) over \(X\) and \(Y\)** evaluate all \(\beta_k\) across the image coherently:

- Independent per-pixel contraction: \(O(WH\cdot n^3)\).
- Grid differences (sketch): about \(O(n^3 + WH\cdot n^2)\) to materialize fibers for all pixels—roughly an \(O(n)\) win on the fiber stage by reusing neighbor state.

### Step D — March in depth with 1D differences

For each pixel, \(\{\beta_k\}\) define a 1D poly in depth. On a **uniform** depth grid, a **1D** difference engine gives:

- \(O(n^2)\) init per ray (or amortized if also differencing in \(Z\) on a full 3D lattice),
- \(O(n)\) additions per depth sample (no Horner muls).

Optional: a full **3D** difference engine on a regular clip lattice \((x,y,z)\) yields roughly \(O(n^3 + M\cdot n^2)\) for \(M\) samples (amortized \(O(n^2)\) per sample) if you voxelize in clip.

### Cost summary (dense degree-\(n\) field)

| Stage | When | Cost (sketch) |
|-------|------|----------------|
| Homogenize | when field changes | once |
| Pullback \(F\to G\) in clip | per view | per-frame, shared |
| 2D Babbage → per-pixel \(\beta_k\) | per view | \(\sim O(n^3+WH\,n^2)\), not \(WH\cdot n^3\) |
| 1D Babbage in depth | per ray / sample | \(O(n^2)+O(n)/\mathrm{sample}\) |
| Dehomogenize / Jacobian | per sample or in measure | see below |

---

## 5. Where to dehomogenize

**Recommendation: keep \(G\) (or \(F\)) homogeneous through the difference engines; apply \(W^{-n}\) (and path Jacobian) at sample time—or fold them into the quadrature weights. Do *not* dehomogenize the coefficient tensor back to Cartesian clip/NDC before grid differences.**

### Why not dehomogenize right after pullback?

Dehomogenizing the **tensor** early means forming Cartesian

\[
g(x,y,z)=G(x,y,z,1)\quad\text{or}\quad f=F/W^n
\]

as a polynomial in NDC \((x,y,z)\). Under perspective, the map from NDC to eye is rational, so a polynomial model in NDC for the **physical** \(f\) is the wrong object (unless you only ever needed \(F\) itself). You either:

- store a rational function, or
- approximate, or
- work in a parameter where the algebra stays polynomial (\(t\), or homogeneous line param).

So early Cartesian dehomogenization into “\(f\) as a poly in \(z_\mathrm{ndc}\)” reintroduces the rational mess Step B avoided.

### Why differences like homogeneous \(G\) / fiber \(\beta_k\)

Forward differences want a **polynomial** on a **uniform grid**. \(G\) (and the \(\beta_k\) as polys in \(x,y\)) satisfy that. Physical \(\sigma \propto \max(0,f)\) with \(f=F/W^n\) is **not** polynomial in the grid coordinates if you bake \(W^{-n}\) into the coeffs.

### Preferred dehomogenization points

**Option 1 — Per sample (simplest to reason about)**  
Difference-engine evaluates \(F\) or \(G\) (or \(\sum\beta_k Z^k W^{n-k}\)) at the sample’s homogeneous/clip coordinates, then:

\[
f=\frac{F}{W^n},\qquad \sigma=\max(0,s\cdot f),
\]

and accumulate \(\tau\) / radiance with the correct Euclidean (or clip) path measure \(ds\).  
Dehomogenization = **\(O(1)\) divide per sample**. Coefficient pipeline stays polynomial end-to-end until here.

**Option 2 — Fold into quadrature weights (same math, fewer explicit divides)**  
Optical depth along a ray:

\[
\tau=\int \sigma\,ds=\int \frac{F}{W^n}\,s\,\left\|\frac{d\mathbf{p}}{d\lambda}\right\|d\lambda.
\]

If \(F(\lambda)\) is the univariate polynomial from the difference engine in parameter \(\lambda\), treat

\[
w(\lambda)=\frac{s}{W(\lambda)^n}\left\|\frac{d\mathbf{p}}{d\lambda}\right\|
\]

as a **known rational weight** (often cheap closed form along a primary ray) and integrate \(F(\lambda)\,w(\lambda)\). You never build a dehomogenized coeff tensor; you dehomogenize in the **measure**.

**Option 3 — Avoid for Path C / Chebyshev-\(T\) prototypes**  
Fitting Chebyshev to \(T=\exp(-\tau)\) can stay in ray-\(t\) or in \(\lambda\) with \(\tau\) from Option 1–2. Don’t convert the whole volume tensor to Cartesian NDC just to match Path C.

### Can we avoid dehomogenizing *per sample*?

Yes. The divide is not fundamental—only “evaluate Cartesian \(f\) from homogeneous \(F\) at this point” needs \(W^{-n}\). Several ways to never do that explicitly:

**A — Analytic / semi-analytic ray integrals (strongest)**  
Along a primary ray, clip/eye \(W(\lambda)=a\lambda+b\) (linear). The difference engine supplies a univariate polynomial \(F(\lambda)\). Then

\[
\int \frac{F(\lambda)}{(a\lambda+b)^n}\,\Big\|\tfrac{d\mathbf{p}}{d\lambda}\Big\|\,d\lambda
\]

is an integral of a **rational function** with known poles. For polynomial \(F\) this has a closed form (partial fractions / reduction), or a stable recurrence. Optical depth and many emission integrals can be computed from **endpoint / antiderivative evaluations of \(F\)'s coeffs**—no marching through dehomogenized samples. Path C can fit \(T=\exp(-\tau)\) from that \(\tau(\lambda)\) directly.

**B — Absorb \(W^{-n}\) into the path measure, integrate \(F\) only**  
Same formula as Option 2. If you still *sample*, you evaluate \(w(\lambda)\) instead of dividing \(F\); if \(w\) is closed form you can also use Gaussian / Chebyshev quadrature on \(F\cdot w\) with nodes that only need **polynomial** \(F\) from Babbage (weight from a formula). The tensor never dehomogenizes; only a scalar weight function does.

**C — Change of parameter with constant weight (rare)**  
Seek \(\mu\) such that \(ds/W^n = c\,d\mu\). Then \(\int f\,ds=c\int F\,d\mu\) and uniform differences in \(\mu\) need no per-sample divide. Under perspective + Euclidean \(ds\), such a \(\mu\) is usually **not** an affine function of clip \(z\) (so you lose the uniform clip grid that made 2D \(X,Y\) Babbage natural). Possible in theory; awkward in practice.

**D — Polynomial approximation of the weight (once per ray)**  
On the ray interval, approximate \(w(\lambda)\approx q(\lambda)\) with \(\deg q=m\) small; form \(F\cdot q\) once (\(O(nm)\)), then pure polynomial marching / differences with **no** dehomogenize. Exactness abandoned for speed/simplicity.

**E — Stay in eye-\(t\) where \(f\) is already polynomial**  
Ray \(\mathbf{p}(t)=t\hat{\mathbf{d}}\) with \(w_{\mathrm{eye}}=1\): Cartesian \(f(t\hat{\mathbf{d}})\) is polynomial in \(t\)—**no \(W^{-n}\) at all**. That sidesteps dehomogenization but also sidesteps the clip-grid story unless fibers are still built in clip and then reparameterized.

**Performance note:** A single divide (or `pow`) per sample is negligible next to fiber construction. Avoiding per-sample dehomogenize matters mainly for **exact Path C / analytic \(\tau\)**, fused quadrature, or keeping the entire inner loop add-only on polynomial state.

**Practical recommendation:** Prefer **A** (analytic \(\tau\) from \(F/(a\lambda+b)^n\)) or **B** (quadrature with closed-form weights) when implementing clip-space differences; keep Option 1 as a debug reference.

### What to avoid

| Approach | Issue |
|----------|--------|
| Dehomogenize tensor → Cartesian NDC poly for \(f\) | Wrong class of function under perspective; breaks “poly on the grid” |
| Dehomogenize before 2D \(X,Y\) differences | Same; loses homogeneous structure that makes fibers polynomials in a clean way |
| Only dehomogenize at Fit (world only forever) | Fine for the *current* world-\(t\) viewer; doesn’t enable clip-grid Babbage |

### Practical default

1. Fit in world → monomials (as now).  
2. On camera change: homogenize (if not stored) → pullback to clip \(G\).  
3. 2D differences on \(\beta_k(x,y)\) over the pixel grid; 1D differences in depth on \(F\) or \(\sum\beta_k Z^k W^{n-k}\).  
4. **Dehomogenize at sample (or in \(ds\) weights)** when forming \(\sigma\) and optical depth.

---

## 6. Relation to the current `poly-cloud` viewer

| Topic | Current viewer | This note |
|-------|----------------|-----------|
| Coeff upload | World monomials once per Fit | Same as Stage 0 |
| Per-view transform | None (no cam pullback) | Homogenize + clip pullback |
| Per pixel | Nested Horner LOS → \(\gamma(u)\) in segment param \(u\) | Screen 2D Babbage → \(\beta_k\); depth 1D Babbage |
| Per sample | Horner on \(\gamma\) | Diff engine + \(W^{-n}\) |
| Path C | Chebyshev \(\hat\tau\) on \(\gamma\), then F2B | Can sit on top of the same samples / \(\tau\) |
| Scaling | \(\Theta(N^3)\) per ray (specialized `FIT_DEG`) | Aim: per-view \(O(n^3+WH n^2)\), march \(O(n)/\mathrm{sample}\) |

Shader `FIT_DEG` specialization only removes the **false max-\(N\) tax** (unrolled deg-8 bodies / huge locals). It does **not** implement clip-grid differences.

---

## 7. Open issues / implementation cautions

1. **Float32 differences** at high degree are unstable (high-order \(\Delta\) amplify cancellation)—seen when Babbage was tried on \(\gamma\) of deg \(\sim 3N\) in the fragment shader. Prefer lower effective \(n\), compensated arithmetic, or double on a compute pass for tables.
2. **Tensor degree \(N\) vs total degree \(n\):** clip pullback of a tensor-product field generally needs total degree up to \(3N\) (or a 4D homogeneous basis sized accordingly).
3. **Frustum / partial tiles:** difference engines want rectangular uniform grids; large empty borders or adaptive resolve need tiling or restart of tables.
4. **Clamping \(\sigma=\max(0,f)\)** breaks pure polynomial marching; differences apply to the poly, clamp after dehomogenization.
5. **Orthographic** is a special case: Cartesian \(f\) is already polynomial in \(z\) at fixed \((x,y)\); homogenization is optional, 2D+1D differences still apply.

---

## 8. One-paragraph abstract (for papers / talks)

*For perspective volume rendering of a dense polynomial density, per-ray restriction of the coefficient tensor costs \(\Omega(N^3)\). Homogenizing the field and pulling it back to clip space makes projection a linear change of basis (once per view). In clip space, primary rays are parallel to depth with pixels on a uniform \((x,y)\) lattice, so the depth-fiber coefficients are bivariate polynomials on that lattice and can be evaluated with 2D forward differences across the image; depth is then marched with a 1D difference engine at \(O(n)\) per sample. Cartesian density is recovered by delaying dehomogenization (\(F/W^n\)) until sample time or folding it into the path-measure weights, preserving a polynomial difference-engine pipeline for as long as possible.*
