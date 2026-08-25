#!/usr/bin/env python3
"""
Validate polynomial transmittance / radiative-transfer approximations
from approx-poly.md against numerical ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_SQRT_2 = np.sqrt(2.0)


def PHI(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal PDF."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def CDF(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF via erf."""
    x_arr = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    out = 0.5 * (1.0 + erf(x_arr / _SQRT_2))
    return float(out) if np.ndim(x) == 0 else out


# Clamped-cubic Φ fit used in Case 2 (approx-poly.md)
M_CUBIC = 2.050


@dataclass
class PolyScene:
    """Polynomial attenuation f and emission g on [z0, zf].

    f(z) = sum_i w[i] (z - z0)^i
    g(z) = sum_j c[j] z^j
    """

    z0: float
    zf: float
    w: np.ndarray  # attenuation coeffs, length deg_f+1
    c: np.ndarray  # emission coeffs, length deg_g+1

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float).ravel()
        self.c = np.asarray(self.c, dtype=float).ravel()
        if self.zf <= self.z0:
            raise ValueError("require zf > z0")


# ---------------------------------------------------------------------------
# Exact quantities
# ---------------------------------------------------------------------------


def f_poly(z: np.ndarray, scene: PolyScene) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    t = z - scene.z0
    out = np.zeros_like(z, dtype=float)
    for i, wi in enumerate(scene.w):
        out += wi * t**i
    return out


def g_poly(z: np.ndarray, scene: PolyScene) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z, dtype=float)
    for j, cj in enumerate(scene.c):
        out += cj * z**j
    return out


def optical_depth_exact(z: np.ndarray, scene: PolyScene) -> np.ndarray:
    """τ(z) = ∫_z^{zf} f = sum_i w_i/((i+1)) [(zf-z0)^{i+1} - (z-z0)^{i+1}]."""
    z = np.asarray(z, dtype=float)
    z0, zf = scene.z0, scene.zf
    tau = np.zeros_like(z, dtype=float)
    for i, wi in enumerate(scene.w):
        tau += wi / (i + 1) * ((zf - z0) ** (i + 1) - (z - z0) ** (i + 1))
    return tau


def T_exact(z: np.ndarray, scene: PolyScene) -> np.ndarray:
    return np.exp(-optical_depth_exact(z, scene))


def T_exact_factorized(z: np.ndarray, scene: PolyScene) -> np.ndarray:
    """A * ∏_i exp(w_i/(i+1) (z-z0)^{i+1}) — algebraically equal to T_exact."""
    z = np.asarray(z, dtype=float)
    z0, zf = scene.z0, scene.zf
    A = 1.0
    prod = np.ones_like(z, dtype=float)
    for i, wi in enumerate(scene.w):
        a_i = wi / (i + 1)
        n = i + 1
        A *= np.exp(-a_i * (zf - z0) ** n)
        prod *= np.exp(a_i * (z - z0) ** n)
    return A * prod


def radiance_exact(scene: PolyScene, n: int = 4001) -> float:
    """∫_{z0}^{zf} g(z) T(z) dz via trapezoid quadrature."""
    z = np.linspace(scene.z0, scene.zf, n)
    return float(np.trapezoid(g_poly(z, scene) * T_exact(z, scene), z))


# ---------------------------------------------------------------------------
# Appendix A: exp(A (x - x_i)^n) ≈ B + C Φ((x-μ)/σ)
# ---------------------------------------------------------------------------


@dataclass
class MonomialApprox:
    A: float
    n: int
    x_i: float
    x_f: float
    B: float
    C: float
    mu: float
    sigma: float
    case: str


def _case1_params(A: float, n: int, x_i: float, x_f: float) -> MonomialApprox:
    """n > 1."""
    if abs(A) < 1e-15:
        return MonomialApprox(A, n, x_i, x_f, B=1.0, C=0.0, mu=x_i, sigma=1.0, case="n>1:A0")
    infl = x_i + (abs((1 - n) / (A * n))) ** (1.0 / n)
    use_infl = (A < 0.0) and (infl < x_f)
    mu = infl if use_infl else x_f
    f_mu = np.exp(A * (mu - x_i) ** n)
    C = 2.0 * (f_mu - 1.0)
    # Slope match; guard near-zero C / A
    denom = A * n * np.sqrt(2.0 * np.pi) * (mu - x_i) ** (n - 1) * np.exp(A * (mu - x_i) ** n)
    if abs(denom) < 1e-30 or abs(C) < 1e-30:
        sigma = (x_f - x_i) / 4.0  # inert fallback
    else:
        sigma = C / denom
        # Equivalent rewritten form in the note:
        # sigma = C/(A n √(2π)) (μ-x_i)^{1-n} exp(-A(μ-x_i)^n)
    return MonomialApprox(A, n, x_i, x_f, B=1.0, C=C, mu=mu, sigma=sigma, case="n>1")


def _c0_from_k(K: float, A: float, delta: float) -> float:
    num = 2.0 * K**3
    den = 1.5 * K + 1.0 - np.exp(A * delta)
    if num * den <= 0.0:
        return float("nan")
    return float(np.sqrt(num / den))


def _case2_params(A: float, x_i: float, x_f: float) -> MonomialApprox:
    """n = 1."""
    delta = x_f - x_i
    if A < 0.0:
        mu = x_i
        K = delta * A * np.sqrt(2.0 * np.pi) / (2.0 * M_CUBIC)
        C0 = _c0_from_k(K, A, delta)
        if np.isnan(C0):
            C = 2.0 * np.exp(A * delta) - 2.0
        elif K < -0.5 * C0:
            C = 2.0 * np.exp(A * delta) - 2.0
        else:
            C = -C0
        B = 1.0 - 0.5 * C
    elif A > 0.0:
        mu = x_f
        K = delta * A * np.sqrt(2.0 * np.pi) / (2.0 * M_CUBIC) * np.exp(A * delta)
        C0 = _c0_from_k(K, A, delta)
        if np.isnan(C0):
            C = 2.0 * np.exp(A * delta) - 2.0
        elif K > 0.5 * C0:
            C = 2.0 * np.exp(A * delta) - 2.0
        else:
            C = C0
        B = np.exp(A * delta) - 0.5 * C
    else:
        # A == 0 → constant 1
        return MonomialApprox(0.0, 1, x_i, x_f, B=1.0, C=0.0, mu=x_i, sigma=1.0, case="n=1:A0")

    sigma = C / (A * np.sqrt(2.0 * np.pi)) * np.exp(-A * (mu - x_i))
    return MonomialApprox(A, 1, x_i, x_f, B=B, C=C, mu=mu, sigma=sigma, case="n=1")


def approx_exp_monomial(A: float, n: int, x_i: float, x_f: float) -> MonomialApprox:
    if n <= 0:
        # Constant exp(A); no z dependence
        return MonomialApprox(A, n, x_i, x_f, B=float(np.exp(A)), C=0.0, mu=x_i, sigma=1.0, case="n=0")
    if n == 1:
        return _case2_params(A, x_i, x_f)
    return _case1_params(A, n, x_i, x_f)


def eval_monomial_approx(z: np.ndarray, p: MonomialApprox) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if abs(p.C) < 1e-30:
        return np.full_like(z, p.B, dtype=float)
    return p.B + p.C * CDF((z - p.mu) / p.sigma)


def exact_exp_monomial(z: np.ndarray, A: float, n: int, x_i: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if n == 0:
        return np.full_like(z, np.exp(A), dtype=float)
    return np.exp(A * (z - x_i) ** n)


# ---------------------------------------------------------------------------
# Transmittance approximation
# ---------------------------------------------------------------------------


@dataclass
class TApproxParts:
    A_scale: float
    factors: list[MonomialApprox]  # unsorted (poly degree order)
    ordered: list[MonomialApprox]  # μ-sorted
    E: np.ndarray
    F: float


def build_T_approx(scene: PolyScene) -> TApproxParts:
    z0, zf = scene.z0, scene.zf
    factors: list[MonomialApprox] = []
    A_scale = 1.0
    for i, wi in enumerate(scene.w):
        a_i = wi / (i + 1)
        n = i + 1
        A_scale *= np.exp(-a_i * (zf - z0) ** n)
        factors.append(approx_exp_monomial(a_i, n, z0, zf))

    order = np.argsort([p.mu for p in factors])
    ordered = [factors[int(k)] for k in order]
    N = len(ordered)
    B = np.array([p.B for p in ordered])
    C = np.array([p.C for p in ordered])
    F = float(np.prod(B)) if N else 1.0
    E = np.zeros(N)
    for i in range(N):
        left = np.prod(B[:i] + C[:i]) if i > 0 else 1.0
        right = np.prod(B[i + 1 :]) if i + 1 < N else 1.0
        E[i] = C[i] * left * right
    return TApproxParts(A_scale=A_scale, factors=factors, ordered=ordered, E=E, F=F)


def T_product_approx(z: np.ndarray, parts: TApproxParts) -> np.ndarray:
    """A * ∏_i (B_i + C_i Φ(h_i)) — Appendix A only, no product collapse."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, parts.A_scale, dtype=float)
    for p in parts.factors:
        out *= eval_monomial_approx(z, p)
    return out


def T_collapsed_approx(z: np.ndarray, parts: TApproxParts) -> np.ndarray:
    """A * (F + sum_i E_i Φ(h_i))."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, parts.F, dtype=float)
    for E_i, p in zip(parts.E, parts.ordered):
        out += E_i * CDF((z - p.mu) / p.sigma)
    return parts.A_scale * out


# ---------------------------------------------------------------------------
# Analytical radiance integral
# ---------------------------------------------------------------------------


def shift_scale_poly_coeffs(c: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """g(z)=sum c_j z^j = sum c'_k ((z-μ)/σ)^k."""
    c = np.asarray(c, dtype=float)
    d = len(c) - 1
    cp = np.zeros(d + 1, dtype=float)
    for j in range(d + 1):
        for k in range(j + 1):
            cp[k] += c[j] * math.comb(j, k) * (sigma**k) * (mu ** (j - k))
    return cp


def M_moments(u0: float, uf: float, k_max: int) -> np.ndarray:
    """M_k = ∫_{u0}^{uf} u^k φ(u) du via recurrence in approx-poly.md."""
    M = np.zeros(k_max + 1, dtype=float)
    if k_max < 0:
        return M
    M[0] = CDF(uf) - CDF(u0)
    if k_max >= 1:
        M[1] = PHI(u0) - PHI(uf)
    for k in range(2, k_max + 1):
        M[k] = (k - 1) * M[k - 2] + (u0 ** (k - 1)) * PHI(u0) - (uf ** (k - 1)) * PHI(uf)
    return M


def integ_uj_Phi(u0: float, uf: float, j: int, sigma: float) -> float:
    """∫ dz ((z-μ)/σ)^j Φ((z-μ)/σ) = σ/(j+1)[u_f^{j+1}Φ_f - u_0^{j+1}Φ_0 - M_{j+1}]."""
    M = M_moments(u0, uf, j + 1)
    return (
        sigma
        / (j + 1)
        * (uf ** (j + 1) * CDF(uf) - u0 ** (j + 1) * CDF(u0) - M[j + 1])
    )


def radiance_approx_analytic(scene: PolyScene, parts: TApproxParts | None = None) -> float:
    if parts is None:
        parts = build_T_approx(scene)
    z0, zf = scene.z0, scene.zf
    # F * ∫ g
    integ_g = 0.0
    for j, cj in enumerate(scene.c):
        integ_g += cj / (j + 1) * (zf ** (j + 1) - z0 ** (j + 1))
    total = parts.F * integ_g

    for E_i, p in zip(parts.E, parts.ordered):
        cp = shift_scale_poly_coeffs(scene.c, p.mu, p.sigma)
        u0 = (z0 - p.mu) / p.sigma
        uf = (zf - p.mu) / p.sigma
        for j, cij in enumerate(cp):
            if abs(cij) < 1e-18:
                continue
            total += E_i * cij * integ_uj_Phi(u0, uf, j, p.sigma)

    return float(parts.A_scale * total)


def radiance_quad_with_T(
    scene: PolyScene, T_fn, n: int = 4001
) -> float:
    z = np.linspace(scene.z0, scene.zf, n)
    return float(np.trapezoid(g_poly(z, scene) * T_fn(z), z))


# ---------------------------------------------------------------------------
# Metrics / suites
# ---------------------------------------------------------------------------


def err_stats(approx: np.ndarray, exact: np.ndarray) -> dict:
    err = approx - exact
    return {
        "max_abs": float(np.max(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rel_l2": float(np.linalg.norm(err) / (np.linalg.norm(exact) + 1e-15)),
        "mean_abs": float(np.mean(np.abs(err))),
    }


def evaluate_scene(scene: PolyScene, n_grid: int = 2001, n_rad: int = 4001) -> dict:
    parts = build_T_approx(scene)
    z = np.linspace(scene.z0, scene.zf, n_grid)
    exact = T_exact(z, scene)
    product = T_product_approx(z, parts)
    collapsed = T_collapsed_approx(z, parts)

    # Per-factor Appendix A error (degree order)
    factor_errs = []
    for i, p in enumerate(parts.factors):
        a_i = scene.w[i] / (i + 1)
        n = i + 1
        ex = exact_exp_monomial(z, a_i, n, scene.z0)
        ap = eval_monomial_approx(z, p)
        factor_errs.append(
            {
                "i": i,
                "n": n,
                "A": a_i,
                "case": p.case,
                "B": p.B,
                "C": p.C,
                "mu": p.mu,
                "sigma": p.sigma,
                **err_stats(ap, ex),
            }
        )

    rad_exact = radiance_exact(scene, n=n_rad)
    rad_prod = radiance_quad_with_T(scene, lambda zz: T_product_approx(zz, parts), n=n_rad)
    rad_col = radiance_quad_with_T(scene, lambda zz: T_collapsed_approx(zz, parts), n=n_rad)
    rad_ana = radiance_approx_analytic(scene, parts)

    f_min = float(np.min(f_poly(z, scene)))

    return {
        "scene": {
            "z0": scene.z0,
            "zf": scene.zf,
            "w": scene.w.tolist(),
            "c": scene.c.tolist(),
            "f_min_on_grid": f_min,
        },
        "T": {
            "product_vs_exact": err_stats(product, exact),
            "collapsed_vs_exact": err_stats(collapsed, exact),
            "collapse_vs_product": err_stats(collapsed, product),
            "endpoint_z0": {
                "exact": float(exact[0]),
                "product": float(product[0]),
                "collapsed": float(collapsed[0]),
            },
            "endpoint_zf": {
                "exact": float(exact[-1]),
                "product": float(product[-1]),
                "collapsed": float(collapsed[-1]),
            },
        },
        "factors": factor_errs,
        "radiance": {
            "exact": rad_exact,
            "quad_product_T": rad_prod,
            "quad_collapsed_T": rad_col,
            "analytic_collapsed": rad_ana,
            "abs_err_analytic": abs(rad_ana - rad_exact),
            "rel_err_analytic": abs(rad_ana - rad_exact) / (abs(rad_exact) + 1e-15),
            "abs_err_quad_collapsed": abs(rad_col - rad_exact),
            "analytic_vs_quad_collapsed": abs(rad_ana - rad_col),
        },
    }


def default_scenes() -> dict[str, PolyScene]:
    """Hand-picked scenes spanning Case 1 / Case 2 and mild polynomials."""
    return {
        "linear_const_f": PolyScene(0.0, 1.0, w=np.array([1.5]), c=np.array([1.0])),
        "linear_emit_poly": PolyScene(0.0, 1.0, w=np.array([1.0]), c=np.array([0.2, 0.5, 0.1])),
        "quadratic_f": PolyScene(0.0, 1.0, w=np.array([0.5, 0.0, 2.0]), c=np.array([1.0])),
        "cubic_f": PolyScene(0.0, 1.0, w=np.array([0.3, 0.2, 0.1, 0.5]), c=np.array([1.0, -0.2])),
        "weak_attenuation": PolyScene(0.0, 2.0, w=np.array([0.1, 0.05]), c=np.array([1.0])),
        "strong_linear": PolyScene(0.0, 1.0, w=np.array([5.0]), c=np.array([1.0])),
        # f stays non-negative on [0,1] for these coeffs
        "mixed_deg4": PolyScene(
            0.0, 1.0, w=np.array([0.4, 0.1, 0.2, 0.0, 0.3]), c=np.array([0.5, 0.5])
        ),
    }


def random_scenes(
    n: int, seed: int = 0, max_deg_f: int = 3, max_deg_g: int = 2
) -> list[tuple[str, PolyScene]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[str, PolyScene]] = []
    for k in range(n):
        z0 = 0.0
        zf = float(rng.uniform(0.5, 2.0))
        df = int(rng.integers(0, max_deg_f + 1))
        dg = int(rng.integers(0, max_deg_g + 1))
        # Non-negative f on [z0,zf]: sample positive coeffs (sufficient for monomials)
        w = rng.uniform(0.05, 1.5, size=df + 1)
        c = rng.normal(0.0, 1.0, size=dg + 1)
        out.append((f"rand_{k}", PolyScene(z0, zf, w, c)))
    return out


def print_summary(name: str, result: dict) -> None:
    T = result["T"]
    R = result["radiance"]
    print(f"\n=== {name} ===")
    print(
        f"  T product   max|e|={T['product_vs_exact']['max_abs']:.3e}  "
        f"rel_L2={T['product_vs_exact']['rel_l2']:.3e}"
    )
    print(
        f"  T collapse  max|e|={T['collapsed_vs_exact']['max_abs']:.3e}  "
        f"rel_L2={T['collapsed_vs_exact']['rel_l2']:.3e}"
    )
    print(
        f"  collapse vs product  max|e|={T['collapse_vs_product']['max_abs']:.3e}"
    )
    print(
        f"  radiance exact={R['exact']:.6g}  analytic={R['analytic_collapsed']:.6g}  "
        f"rel_err={R['rel_err_analytic']:.3e}  "
        f"|ana-quad_col|={R['analytic_vs_quad_collapsed']:.3e}"
    )
    print(f"  f_min={result['scene']['f_min_on_grid']:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["default", "random", "all"], default="default")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "poly_validation_results.json")
    ap.add_argument("--grid", type=int, default=2001)
    args = ap.parse_args()

    scenes: list[tuple[str, PolyScene]] = []
    if args.suite in ("default", "all"):
        scenes.extend(default_scenes().items())
    if args.suite in ("random", "all"):
        scenes.extend(random_scenes(args.n_random, seed=args.seed))

    results = {}
    for name, scene in scenes:
        results[name] = evaluate_scene(scene, n_grid=args.grid)
        print_summary(name, results[name])

    # Aggregate
    rel_T = [r["T"]["collapsed_vs_exact"]["rel_l2"] for r in results.values()]
    rel_R = [r["radiance"]["rel_err_analytic"] for r in results.values()]
    summary = {
        "n_scenes": len(results),
        "T_collapsed_rel_l2": {
            "mean": float(np.mean(rel_T)),
            "max": float(np.max(rel_T)),
            "median": float(np.median(rel_T)),
        },
        "radiance_analytic_rel_err": {
            "mean": float(np.mean(rel_R)),
            "max": float(np.max(rel_R)),
            "median": float(np.median(rel_R)),
        },
    }
    print("\n=== aggregate ===")
    print(json.dumps(summary, indent=2))

    payload = {"summary": summary, "scenes": results}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
