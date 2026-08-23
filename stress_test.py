#!/usr/bin/env python3
"""
Stress-test the multi-Gaussian transmittance / radiance approximation
for larger N, and decompose error into Appendix A vs product-collapse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validate_approx import (
    CDF,
    Gaussians,
    T_approx,
    T_error,
    T_exact,
    appendix_a_params,
    radiance_approx,
    radiance_exact,
    sample_curve,
)


def T_product_appendix_a(z: np.ndarray, zf: float, g: Gaussians) -> np.ndarray:
    """A * ∏_i (1+(e^{w_i}-1) Φ(h_i)) — Appendix A only, no product collapse."""
    g = g.sorted()
    z = np.asarray(z, dtype=float)
    A = np.exp(-np.sum(g.w * CDF((zf - g.mu) / g.sigma)))
    delta, nu = appendix_a_params(g.w, g.sigma)
    h = (z[..., None] - g.mu - delta) / nu
    factors = 1.0 + (np.exp(g.w) - 1.0) * CDF(h)
    return A * np.prod(factors, axis=-1)


def T_error_decomposed(zi: float, zf: float, g: Gaussians, n: int = 2001) -> dict:
    z = np.linspace(zi, zf, n)
    exact = T_exact(z, zf, g)
    product = T_product_appendix_a(z, zf, g)
    collapsed = T_approx(z, zf, g)

    def stats(approx: np.ndarray) -> dict:
        err = approx - exact
        return {
            "max_abs": float(np.max(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "rel_l2": float(np.linalg.norm(err) / (np.linalg.norm(exact) + 1e-15)),
        }

    collapse_only = collapsed - product
    return {
        "vs_exact_product": stats(product),
        "vs_exact_collapsed": stats(collapsed),
        "collapse_vs_product": {
            "max_abs": float(np.max(np.abs(collapse_only))),
            "rmse": float(np.sqrt(np.mean(collapse_only**2))),
            "rel_l2": float(
                np.linalg.norm(collapse_only) / (np.linalg.norm(product) + 1e-15)
            ),
        },
        "endpoint_zi": {
            "exact": float(exact[0]),
            "product": float(product[0]),
            "collapsed": float(collapsed[0]),
        },
        "endpoint_zf": {
            "exact": float(exact[-1]),
            "product": float(product[-1]),
            "collapsed": float(collapsed[-1]),
        },
    }


def scene_separated(n: int, w: float = 0.5, sigma: float = 0.25) -> Gaussians:
    """Means evenly spaced on a fixed [0.5, 9.5] interval (overlap grows with N)."""
    mu = np.linspace(0.5, 9.5, n)
    return Gaussians(
        mu=mu,
        sigma=np.full(n, sigma),
        w=np.full(n, w),
        c=np.full(n, w),
    )


def scene_fixed_separation(
    n: int, w: float = 0.5, sigma: float = 0.25, gap_sigmas: float = 4.0
) -> tuple[Gaussians, float, float]:
    """Keep Δμ/σ fixed so neighbors barely overlap; domain grows with N."""
    spacing = gap_sigmas * sigma
    mu = np.arange(n, dtype=float) * spacing + 2.0
    g = Gaussians(
        mu=mu,
        sigma=np.full(n, sigma),
        w=np.full(n, w),
        c=np.full(n, w),
    )
    return g, -1.0, float(mu[-1] + 2.0)


def scene_clustered(n: int, w: float = 0.5, sigma: float = 0.35) -> Gaussians:
    """All means packed into a short depth interval."""
    mu = np.linspace(3.0, 5.0, n)
    return Gaussians(
        mu=mu,
        sigma=np.full(n, sigma),
        w=np.full(n, w),
        c=np.full(n, w),
    )


def scene_random(
    n: int,
    seed: int,
    z_lo: float = 0.0,
    z_hi: float = 10.0,
    w_range: tuple[float, float] = (0.2, 1.2),
    sigma_range: tuple[float, float] = (0.15, 0.5),
) -> Gaussians:
    rng = np.random.default_rng(seed)
    mu = np.sort(rng.uniform(z_lo, z_hi, n))
    sigma = rng.uniform(sigma_range[0], sigma_range[1], n)
    w = rng.uniform(w_range[0], w_range[1], n)
    c = rng.uniform(0.3, 1.0, n)
    return Gaussians(mu=mu, sigma=sigma, w=w, c=c)


def evaluate_scene(
    name: str,
    g: Gaussians,
    zi: float,
    zf: float,
    *,
    do_radiance: bool = True,
) -> dict:
    decomp = T_error_decomposed(zi, zf, g)
    row: dict = {
        "name": name,
        "N": int(len(g.mu)),
        "total_optical_weight": float(np.sum(g.w)),
        "mean_w": float(np.mean(g.w)),
        "T": decomp,
    }
    if do_radiance:
        r_ex = radiance_exact(zi, zf, g)
        r_ap = radiance_approx(zi, zf, g)
        row["radiance"] = {
            "exact": r_ex,
            "approx": r_ap,
            "abs_err": abs(r_ap - r_ex),
            "rel_err": abs(r_ap - r_ex) / (abs(r_ex) + 1e-15),
        }
    return row


def main() -> None:
    zi, zf = -2.0, 12.0
    ns = [2, 4, 8, 16, 32, 64]

    results: dict = {
        "interval": {"zi": zi, "zf": zf},
        "n_scaling": {
            "separated": [],
            "clustered": [],
            "separated_strong": [],
            "fixed_separation": [],
        },
        "random": [],
        "curves": {},
    }

    print("=== N scaling: separated (w=0.5, σ=0.25) ===")
    for n in ns:
        g = scene_separated(n, w=0.5, sigma=0.25)
        # Skip radiance for N>32 — O(N²) bivariate CDFs get slow
        row = evaluate_scene(
            f"sep_n{n}", g, zi, zf, do_radiance=(n <= 32)
        )
        results["n_scaling"]["separated"].append(row)
        rad = row.get("radiance", {})
        print(
            f"  N={n:3d}  Σw={row['total_optical_weight']:.1f}  "
            f"T_prod={row['T']['vs_exact_product']['rel_l2']:.3e}  "
            f"T_full={row['T']['vs_exact_collapsed']['rel_l2']:.3e}  "
            f"collapse={row['T']['collapse_vs_product']['rel_l2']:.3e}  "
            f"rad_rel={rad.get('rel_err', float('nan')):.3e}"
        )

    print("\n=== N scaling: clustered in [3,5] (w=0.5, σ=0.35) ===")
    for n in ns:
        g = scene_clustered(n, w=0.5, sigma=0.35)
        row = evaluate_scene(
            f"clu_n{n}", g, zi, zf, do_radiance=(n <= 32)
        )
        results["n_scaling"]["clustered"].append(row)
        rad = row.get("radiance", {})
        print(
            f"  N={n:3d}  Σw={row['total_optical_weight']:.1f}  "
            f"T_prod={row['T']['vs_exact_product']['rel_l2']:.3e}  "
            f"T_full={row['T']['vs_exact_collapsed']['rel_l2']:.3e}  "
            f"collapse={row['T']['collapse_vs_product']['rel_l2']:.3e}  "
            f"rad_rel={rad.get('rel_err', float('nan')):.3e}"
        )

    print("\n=== N scaling: separated strong (w=1.0, σ=0.25) ===")
    for n in ns:
        g = scene_separated(n, w=1.0, sigma=0.25)
        row = evaluate_scene(
            f"sep_strong_n{n}", g, zi, zf, do_radiance=(n <= 16)
        )
        results["n_scaling"]["separated_strong"].append(row)
        rad = row.get("radiance", {})
        print(
            f"  N={n:3d}  Σw={row['total_optical_weight']:.1f}  "
            f"T_prod={row['T']['vs_exact_product']['rel_l2']:.3e}  "
            f"T_full={row['T']['vs_exact_collapsed']['rel_l2']:.3e}  "
            f"collapse={row['T']['collapse_vs_product']['rel_l2']:.3e}  "
            f"rad_rel={rad.get('rel_err', float('nan')):.3e}"
        )

    print("\n=== N scaling: fixed Δμ=4σ (domain grows with N) ===")
    for n in ns:
        g, zi_fs, zf_fs = scene_fixed_separation(n, w=0.5, sigma=0.25)
        row = evaluate_scene(
            f"fix_sep_n{n}", g, zi_fs, zf_fs, do_radiance=(n <= 16)
        )
        results["n_scaling"]["fixed_separation"].append(row)
        rad = row.get("radiance", {})
        print(
            f"  N={n:3d}  domain=[{zi_fs:.0f},{zf_fs:.1f}]  Σw={row['total_optical_weight']:.1f}  "
            f"T_prod={row['T']['vs_exact_product']['rel_l2']:.3e}  "
            f"T_full={row['T']['vs_exact_collapsed']['rel_l2']:.3e}  "
            f"collapse={row['T']['collapse_vs_product']['rel_l2']:.3e}  "
            f"rad_rel={rad.get('rel_err', float('nan')):.3e}"
        )

    print("\n=== Random mixtures (5 seeds × N) ===")
    for n in [8, 16, 32, 64]:
        seed_rows = []
        for seed in range(5):
            g = scene_random(n, seed=seed)
            row = evaluate_scene(
                f"rand_n{n}_s{seed}",
                g,
                zi,
                zf,
                do_radiance=(n <= 16),
            )
            seed_rows.append(row)
        t_full = [r["T"]["vs_exact_collapsed"]["rel_l2"] for r in seed_rows]
        t_prod = [r["T"]["vs_exact_product"]["rel_l2"] for r in seed_rows]
        collapse = [r["T"]["collapse_vs_product"]["rel_l2"] for r in seed_rows]
        rads = [
            r["radiance"]["rel_err"] for r in seed_rows if "radiance" in r
        ]
        summary = {
            "N": n,
            "seeds": 5,
            "T_full_rel_l2": {
                "mean": float(np.mean(t_full)),
                "std": float(np.std(t_full)),
                "max": float(np.max(t_full)),
            },
            "T_product_rel_l2": {
                "mean": float(np.mean(t_prod)),
                "std": float(np.std(t_prod)),
                "max": float(np.max(t_prod)),
            },
            "collapse_rel_l2": {
                "mean": float(np.mean(collapse)),
                "std": float(np.std(collapse)),
                "max": float(np.max(collapse)),
            },
            "rows": seed_rows,
        }
        if rads:
            summary["rad_rel_err"] = {
                "mean": float(np.mean(rads)),
                "std": float(np.std(rads)),
                "max": float(np.max(rads)),
            }
        results["random"].append(summary)
        rad_msg = (
            f"  rad_mean={summary['rad_rel_err']['mean']:.3e}"
            if rads
            else "  rad=skipped"
        )
        print(
            f"  N={n:3d}  T_full mean/max={summary['T_full_rel_l2']['mean']:.3e}/"
            f"{summary['T_full_rel_l2']['max']:.3e}  "
            f"T_prod mean={summary['T_product_rel_l2']['mean']:.3e}  "
            f"collapse mean={summary['collapse_rel_l2']['mean']:.3e}"
            f"{rad_msg}"
        )

    # Representative curves for the canvas
    for label, g in [
        ("sep_n16", scene_separated(16, w=0.5, sigma=0.25)),
        ("clu_n16", scene_clustered(16, w=0.5, sigma=0.35)),
        ("sep_n64", scene_separated(64, w=0.5, sigma=0.25)),
        ("clu_n64", scene_clustered(64, w=0.5, sigma=0.35)),
        ("rand_n32_s0", scene_random(32, seed=0)),
    ]:
        z = np.linspace(zi, zf, 400)
        results["curves"][label] = {
            "z": z.tolist(),
            "T_exact": T_exact(z, zf, g).tolist(),
            "T_product": T_product_appendix_a(z, zf, g).tolist(),
            "T_collapsed": T_approx(z, zf, g).tolist(),
            "N": int(len(g.mu)),
            "total_optical_weight": float(np.sum(g.w)),
        }

    out_path = Path(__file__).with_name("stress_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
