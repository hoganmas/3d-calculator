#!/usr/bin/env python3
"""
Three-way blending benchmark vs numerical RTE ground truth.

  A) Classic discrete α-blend  (α=1-e^{-w}, front-to-back from camera at zf)
  B) Exact T(z) + Gauss-Hermite emission integral
  C) Collapsed analytical radiance (Appendix A + Φ-product collapse + Φ₂)
  GT) High-res trapezoidal ∫ g(z) T_exact(z) dz
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_hermitenorm

from stress_test import (
    scene_clustered,
    scene_fixed_separation,
    scene_random,
    scene_separated,
)
from validate_approx import (
    CDF,
    Gaussians,
    T_exact,
    emission,
    make_scene,
    radiance_approx,
    radiance_exact,
)


def classic_alpha_blend(g: Gaussians, zi: float, zf: float) -> float:
    """
    Discrete front-to-back α compositing.

    Camera exits at zf (looking toward decreasing z). Front = larger μ.
    Maps continuous (extinction w, emission c) to splat opacity/albedo:
      α_i = 1 - exp(-w_i),  albedo_i = c_i / w_i
      C += albedo · α · T,  T *= (1-α)
    which matches ∫ c φ T for an isolated fully-covered Gaussian (→ albedo(1-e^{-w})).
    """
    g = g.sorted()
    order = np.argsort(-g.mu)
    T = 1.0
    C = 0.0
    for i in order:
        if g.mu[i] < zi - 4 * g.sigma[i] or g.mu[i] > zf + 4 * g.sigma[i]:
            continue
        w = float(g.w[i])
        if w < 1e-15:
            continue
        alpha = 1.0 - np.exp(-w)
        albedo = float(g.c[i]) / w
        C += albedo * alpha * T
        T *= 1.0 - alpha
        if T < 1e-6:
            break
    return float(C)


def optical_depth_front_blend(g: Gaussians, zi: float, zf: float) -> float:
    """
    O(N²) physics-ish drop-in: albedo·α attenuated by exact front optical depth
    τ_i = Σ_{j: μ_j > μ_i} w_j (Φ_j(zf) - Φ_j(μ_i)).
    """
    g = g.sorted()
    C = 0.0
    for i in range(len(g.mu)):
        w = float(g.w[i])
        if w < 1e-15:
            continue
        tau = 0.0
        for j in range(len(g.mu)):
            if g.mu[j] <= g.mu[i]:
                continue
            tau += g.w[j] * (
                CDF((zf - g.mu[j]) / g.sigma[j]) - CDF((g.mu[i] - g.mu[j]) / g.sigma[j])
            )
        alpha = 1.0 - np.exp(-w)
        albedo = float(g.c[i]) / w
        C += albedo * alpha * np.exp(-tau)
    return float(C)


def exact_T_hermite(
    g: Gaussians, zi: float, zf: float, n_nodes: int = 8
) -> float:
    """
    I ≈ Σ_j c_j ∫ φ_σj(z) T_exact(z) dz
      = Σ_j c_j ∫ φ(u) T(μ_j + σ_j u) du

    scipy roots_hermitenorm integrates ∫ e^{-u²/2} f(u) du, so divide weights
    by √(2π) to get the standard-normal measure.
    """
    g = g.sorted()
    x, wgh = roots_hermitenorm(n_nodes)
    wgh = wgh / np.sqrt(2.0 * np.pi)
    C = 0.0
    for mu, sigma, c in zip(g.mu, g.sigma, g.c):
        zs = mu + sigma * x
        mask = (zs >= zi) & (zs <= zf)
        if not np.any(mask):
            continue
        Tz = T_exact(zs[mask], zf, g)
        C += c * float(np.sum(wgh[mask] * Tz))
    return float(C)


def rel_err(approx: float, exact: float) -> float:
    return abs(approx - exact) / (abs(exact) + 1e-15)


def evaluate_methods(
    g: Gaussians,
    zi: float,
    zf: float,
    *,
    do_analytic: bool = True,
    hermite_nodes: int = 8,
) -> dict:
    g = g.sorted()
    t0 = time.perf_counter()
    gt = radiance_exact(zi, zf, g)
    t_gt = time.perf_counter() - t0

    t0 = time.perf_counter()
    a = classic_alpha_blend(g, zi, zf)
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    od = optical_depth_front_blend(g, zi, zf)
    t_od = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = exact_T_hermite(g, zi, zf, n_nodes=hermite_nodes)
    t_b = time.perf_counter() - t0

    row: dict = {
        "N": int(len(g.mu)),
        "total_w": float(np.sum(g.w)),
        "gt": gt,
        "classic_alpha": {"value": a, "rel_err": rel_err(a, gt), "ms": t_a * 1e3},
        "od_front": {"value": od, "rel_err": rel_err(od, gt), "ms": t_od * 1e3},
        "exact_T_hermite": {
            "value": b,
            "rel_err": rel_err(b, gt),
            "ms": t_b * 1e3,
            "nodes": hermite_nodes,
        },
        "gt_ms": t_gt * 1e3,
    }

    if do_analytic:
        t0 = time.perf_counter()
        c = radiance_approx(zi, zf, g)
        t_c = time.perf_counter() - t0
        row["analytic_collapsed"] = {
            "value": c,
            "rel_err": rel_err(c, gt),
            "ms": t_c * 1e3,
        }
    return row


def fmt(row: dict) -> str:
    parts = [
        f"N={row['N']:3d}",
        f"α={row['classic_alpha']['rel_err']:.3e}",
        f"od={row['od_front']['rel_err']:.3e}",
        f"H={row['exact_T_hermite']['rel_err']:.3e}",
    ]
    if "analytic_collapsed" in row:
        parts.append(f"A={row['analytic_collapsed']['rel_err']:.3e}")
    parts.append(f"gt={row['gt']:.4f}")
    return "  ".join(parts)


def main() -> None:
    results: dict = {"scenes": [], "n_scaling": {}, "hermite_convergence": [], "random": []}

    print("=== Hand scenes ===")
    zi, zf = -4.0, 8.0
    hand = [
        make_scene("single_weak", [2.0], [0.5], [0.3]),
        make_scene("single_moderate", [2.0], [0.5], [1.0]),
        make_scene("single_strong", [2.0], [0.5], [3.0]),
        make_scene("two_separated", [1.0, 5.0], [0.4, 0.4], [1.0, 1.0]),
        make_scene("two_heavy_overlap", [2.0, 2.3], [0.6, 0.6], [1.2, 1.2]),
        make_scene("three_clustered", [2.0, 2.5, 3.0], [0.45, 0.45, 0.45], [1.0, 1.0, 1.0]),
        make_scene(
            "unequal_weights",
            [1.0, 3.0, 5.5],
            [0.4, 0.5, 0.3],
            [0.4, 2.5, 0.7],
            [1.0, 1.0, 1.0],
        ),
        (
            "overlap_diff_albedo",
            Gaussians(
                mu=np.array([2.0, 2.4]),
                sigma=np.array([0.5, 0.5]),
                w=np.array([1.0, 1.0]),
                c=np.array([0.2, 2.0]),
            ),
        ),
        (
            "sep_diff_albedo",
            Gaussians(
                mu=np.array([1.0, 5.0]),
                sigma=np.array([0.4, 0.4]),
                w=np.array([1.0, 1.0]),
                c=np.array([0.2, 2.0]),
            ),
        ),
    ]
    for name, g in hand:
        row = evaluate_methods(g, zi, zf)
        row["name"] = name
        results["scenes"].append(row)
        print(f"  {name:22s}  {fmt(row)}")

    print("\n=== Hermite node convergence (two_heavy_overlap) ===")
    _, g_ov = make_scene("ov", [2.0, 2.3], [0.6, 0.6], [1.2, 1.2])
    gt = radiance_exact(zi, zf, g_ov)
    for n in [2, 4, 6, 8, 12, 16]:
        val = exact_T_hermite(g_ov, zi, zf, n_nodes=n)
        results["hermite_convergence"].append(
            {"nodes": n, "value": val, "rel_err": rel_err(val, gt), "gt": gt}
        )
        print(f"  nodes={n:2d}  rel={rel_err(val, gt):.3e}  val={val:.6f}  gt={gt:.6f}")

    print("\n=== N scaling: fixed Δμ=4σ ===")
    results["n_scaling"]["fixed_sep"] = []
    for n in [2, 4, 8, 16, 32]:
        g, zi_fs, zf_fs = scene_fixed_separation(n, w=0.5, sigma=0.25)
        row = evaluate_methods(g, zi_fs, zf_fs, do_analytic=(n <= 16))
        row["name"] = f"fix_sep_n{n}"
        results["n_scaling"]["fixed_sep"].append(row)
        print(f"  {fmt(row)}")

    print("\n=== N scaling: clustered [3,5] ===")
    results["n_scaling"]["clustered"] = []
    zi_c, zf_c = -2.0, 12.0
    for n in [2, 4, 8, 16, 32]:
        g = scene_clustered(n, w=0.5, sigma=0.35)
        row = evaluate_methods(g, zi_c, zf_c, do_analytic=(n <= 16))
        row["name"] = f"clu_n{n}"
        results["n_scaling"]["clustered"].append(row)
        print(f"  {fmt(row)}")

    print("\n=== N scaling: packed on [0.5,9.5] ===")
    results["n_scaling"]["packed"] = []
    for n in [2, 4, 8, 16, 32]:
        g = scene_separated(n, w=0.5, sigma=0.25)
        row = evaluate_methods(g, zi_c, zf_c, do_analytic=(n <= 16))
        row["name"] = f"pack_n{n}"
        results["n_scaling"]["packed"].append(row)
        print(f"  {fmt(row)}")

    print("\n=== N scaling: clustered, albedo ≠ 1 (c=1, w=0.5) ===")
    results["n_scaling"]["clustered_albedo"] = []
    for n in [2, 4, 8, 16, 32]:
        g = scene_clustered(n, w=0.5, sigma=0.35)
        g = Gaussians(mu=g.mu, sigma=g.sigma, w=g.w, c=np.ones(n))
        row = evaluate_methods(g, zi_c, zf_c, do_analytic=(n <= 16))
        row["name"] = f"clu_alb_n{n}"
        results["n_scaling"]["clustered_albedo"].append(row)
        print(f"  {fmt(row)}")

    print("\n=== N scaling: fixed sep, albedo ≠ 1 (c=1, w=0.5) ===")
    results["n_scaling"]["fixed_sep_albedo"] = []
    for n in [2, 4, 8, 16, 32]:
        g, zi_fs, zf_fs = scene_fixed_separation(n, w=0.5, sigma=0.25)
        g = Gaussians(mu=g.mu, sigma=g.sigma, w=g.w, c=np.ones(n))
        row = evaluate_methods(g, zi_fs, zf_fs, do_analytic=(n <= 16))
        row["name"] = f"fix_sep_alb_n{n}"
        results["n_scaling"]["fixed_sep_albedo"].append(row)
        print(f"  {fmt(row)}")

    print("\n=== Random (5 seeds) ===")
    for n in [8, 16, 32]:
        rows = []
        for seed in range(5):
            g = scene_random(n, seed=seed)
            row = evaluate_methods(g, zi_c, zf_c, do_analytic=(n <= 16))
            rows.append(row)
        summary = {
            "N": n,
            "classic_alpha": {
                "mean": float(np.mean([r["classic_alpha"]["rel_err"] for r in rows])),
                "max": float(np.max([r["classic_alpha"]["rel_err"] for r in rows])),
            },
            "od_front": {
                "mean": float(np.mean([r["od_front"]["rel_err"] for r in rows])),
                "max": float(np.max([r["od_front"]["rel_err"] for r in rows])),
            },
            "exact_T_hermite": {
                "mean": float(np.mean([r["exact_T_hermite"]["rel_err"] for r in rows])),
                "max": float(np.max([r["exact_T_hermite"]["rel_err"] for r in rows])),
            },
            "rows": rows,
        }
        if n <= 16:
            summary["analytic_collapsed"] = {
                "mean": float(
                    np.mean([r["analytic_collapsed"]["rel_err"] for r in rows])
                ),
                "max": float(
                    np.max([r["analytic_collapsed"]["rel_err"] for r in rows])
                ),
            }
        results["random"].append(summary)
        a = summary["classic_alpha"]
        h = summary["exact_T_hermite"]
        o = summary["od_front"]
        msg = (
            f"  N={n:3d}  α mean/max={a['mean']:.3e}/{a['max']:.3e}  "
            f"od={o['mean']:.3e}/{o['max']:.3e}  "
            f"H={h['mean']:.3e}/{h['max']:.3e}"
        )
        if "analytic_collapsed" in summary:
            ac = summary["analytic_collapsed"]
            msg += f"  A={ac['mean']:.3e}/{ac['max']:.3e}"
        print(msg)

    # Timing microbench on clustered N=16
    print("\n=== Timing (clustered N=16, 20 runs) ===")
    g = scene_clustered(16, w=0.5, sigma=0.35)
    timings = {}
    for label, fn in [
        ("classic_alpha", lambda: classic_alpha_blend(g, zi_c, zf_c)),
        ("od_front", lambda: optical_depth_front_blend(g, zi_c, zf_c)),
        ("exact_T_hermite8", lambda: exact_T_hermite(g, zi_c, zf_c, 8)),
        ("analytic_collapsed", lambda: radiance_approx(zi_c, zf_c, g)),
        ("gt_trap8192", lambda: radiance_exact(zi_c, zf_c, g)),
    ]:
        # warmup
        fn()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        timings[label] = {
            "mean_ms": float(np.mean(ts) * 1e3),
            "std_ms": float(np.std(ts) * 1e3),
        }
        print(f"  {label:20s}  {timings[label]['mean_ms']:.3f} ± {timings[label]['std_ms']:.3f} ms")
    results["timings_clu16"] = timings

    out = Path(__file__).resolve().parent / "results" / "blend_benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
