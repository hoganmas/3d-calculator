#!/usr/bin/env python3
"""
Performance benchmarks for blending / RTE integrators.

Reports wall time vs N (mean ± std over repeats) for:
  - classic α-blend
  - exact T + Gauss-Hermite (4 / 8 nodes)
  - batched Hermite (one T_exact over all sample depths)
  - collapsed analytic (N ≤ 16)
  - trapezoidal GT
  - cluster_composite (Hermite / analytic-n2)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_hermitenorm

from blend_benchmark import classic_alpha_blend, exact_T_hermite
from cluster_blend import cluster_composite
from stress_test import scene_clustered, scene_fixed_separation, scene_random
from validate_approx import T_exact, Gaussians, radiance_approx, radiance_exact


def bench(fn, repeats: int = 25, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    arr = np.asarray(ts)
    return {
        "mean_ms": float(np.mean(arr) * 1e3),
        "std_ms": float(np.std(arr) * 1e3),
        "min_ms": float(np.min(arr) * 1e3),
        "repeats": repeats,
    }


def exact_T_hermite_batched(
    g: Gaussians, zi: float, zf: float, n_nodes: int = 8
) -> float:
    """
    Same estimator as exact_T_hermite, but evaluates T_exact once on the
    concatenated node depths (less Python overhead; still O(N²·nodes) work).
    """
    g = g.sorted()
    x, wgh = roots_hermitenorm(n_nodes)
    wgh = wgh / np.sqrt(2.0 * np.pi)

    zs_list = []
    meta = []  # (c, slice)
    for mu, sigma, c in zip(g.mu, g.sigma, g.c):
        zs = mu + sigma * x
        mask = (zs >= zi) & (zs <= zf)
        if not np.any(mask):
            continue
        start = len(zs_list)
        zs_list.extend(zs[mask].tolist())
        meta.append((float(c), wgh[mask], start, len(zs_list)))

    if not zs_list:
        return 0.0
    z_all = np.asarray(zs_list, dtype=float)
    T_all = T_exact(z_all, zf, g)
    C = 0.0
    for c, w, a, b in meta:
        C += c * float(np.sum(w * T_all[a:b]))
    return float(C)


def main() -> None:
    zi, zf = -2.0, 12.0
    ns = [1, 2, 4, 8, 16, 32, 64]
    out: dict = {
        "note": (
            "CPython timings on synthetic 1D mixtures. Hermite cost is "
            "O(N_lobes · N_extinction) ≈ O(N²) with exact T per sample."
        ),
        "clustered": [],
        "fixed_sep": [],
        "nodes_sweep": [],
        "cluster_router": [],
    }

    print("=== Clustered packing [3,5] — time vs N ===")
    print(
        f"{'N':>4} {'α':>9} {'H4':>9} {'H8':>9} {'H8bat':>9} "
        f"{'Analytic':>9} {'Trap4k':>9}"
    )
    for n in ns:
        g = scene_clustered(n, w=0.5, sigma=0.35)
        repeats = 40 if n <= 8 else (20 if n <= 32 else 10)
        row = {"N": n, "layout": "clustered"}
        row["classic_alpha"] = bench(
            lambda: classic_alpha_blend(g, zi, zf), repeats=repeats
        )
        row["hermite4"] = bench(
            lambda: exact_T_hermite(g, zi, zf, 4), repeats=repeats
        )
        row["hermite8"] = bench(
            lambda: exact_T_hermite(g, zi, zf, 8), repeats=repeats
        )
        row["hermite8_batched"] = bench(
            lambda: exact_T_hermite_batched(g, zi, zf, 8), repeats=repeats
        )
        if n <= 16:
            row["analytic"] = bench(
                lambda: radiance_approx(zi, zf, g), repeats=max(10, repeats // 2)
            )
        row["trap4096"] = bench(
            lambda: radiance_exact(zi, zf, g, n=4096), repeats=repeats
        )
        out["clustered"].append(row)

        def fmt(key: str) -> str:
            if key not in row:
                return f"{'—':>9}"
            return f"{row[key]['mean_ms']:9.3f}"

        print(
            f"{n:4d} {fmt('classic_alpha')} {fmt('hermite4')} {fmt('hermite8')} "
            f"{fmt('hermite8_batched')} {fmt('analytic')} {fmt('trap4096')}"
        )

    print("\n=== Fixed Δμ=4σ — time vs N (Hermite should stay easier / similar) ===")
    print(f"{'N':>4} {'α':>9} {'H8':>9} {'H8bat':>9} {'Trap4k':>9}")
    for n in ns:
        g, zi_fs, zf_fs = scene_fixed_separation(n, w=0.5, sigma=0.25)
        repeats = 30 if n <= 16 else 10
        row = {"N": n, "layout": "fixed_sep", "domain": [zi_fs, zf_fs]}
        row["classic_alpha"] = bench(
            lambda: classic_alpha_blend(g, zi_fs, zf_fs), repeats=repeats
        )
        row["hermite8"] = bench(
            lambda: exact_T_hermite(g, zi_fs, zf_fs, 8), repeats=repeats
        )
        row["hermite8_batched"] = bench(
            lambda: exact_T_hermite_batched(g, zi_fs, zf_fs, 8), repeats=repeats
        )
        row["trap4096"] = bench(
            lambda: radiance_exact(zi_fs, zf_fs, g, n=4096), repeats=repeats
        )
        out["fixed_sep"].append(row)
        print(
            f"{n:4d} {row['classic_alpha']['mean_ms']:9.3f} "
            f"{row['hermite8']['mean_ms']:9.3f} "
            f"{row['hermite8_batched']['mean_ms']:9.3f} "
            f"{row['trap4096']['mean_ms']:9.3f}"
        )

    print("\n=== Hermite node sweep (clustered N=16) ===")
    g16 = scene_clustered(16, w=0.5, sigma=0.35)
    for nodes in [2, 4, 6, 8, 12, 16]:
        row = {
            "N": 16,
            "nodes": nodes,
            "hermite": bench(
                lambda n=nodes: exact_T_hermite(g16, zi, zf, n), repeats=20
            ),
            "hermite_batched": bench(
                lambda n=nodes: exact_T_hermite_batched(g16, zi, zf, n), repeats=20
            ),
        }
        out["nodes_sweep"].append(row)
        print(
            f"  nodes={nodes:2d}  H={row['hermite']['mean_ms']:.3f} ms  "
            f"Hbat={row['hermite_batched']['mean_ms']:.3f} ms"
        )

    print("\n=== Cluster router vs global (random N=16, 5 seeds) ===")
    for seed in range(5):
        g = scene_random(16, seed=seed)
        row = {
            "seed": seed,
            "global_hermite8": bench(
                lambda: exact_T_hermite(g, zi, zf, 8), repeats=15
            ),
            "cluster_hermite": bench(
                lambda: cluster_composite(g, zi, zf, method="hermite")[0],
                repeats=15,
            ),
            "cluster_analytic_n2": bench(
                lambda: cluster_composite(
                    g, zi, zf, method="analytic", max_analytic_n=2
                )[0],
                repeats=15,
            ),
            "classic_alpha": bench(
                lambda: classic_alpha_blend(g, zi, zf), repeats=30
            ),
        }
        # cluster stats
        _, st, _ = cluster_composite(g, zi, zf, method="hermite")
        row["clusters"] = {
            "n": st.n_clusters,
            "sizes": st.sizes,
            "pair": st.n_pair,
            "large": st.n_large,
        }
        out["cluster_router"].append(row)
        print(
            f"  seed={seed}  sizes={st.sizes}  "
            f"gH={row['global_hermite8']['mean_ms']:.2f}  "
            f"cH={row['cluster_hermite']['mean_ms']:.2f}  "
            f"cA2={row['cluster_analytic_n2']['mean_ms']:.2f}  "
            f"α={row['classic_alpha']['mean_ms']:.3f} ms"
        )

    # Complexity summary from clustered H8
    h8 = [r["hermite8"]["mean_ms"] for r in out["clustered"]]
    ns_a = [r["N"] for r in out["clustered"]]
    # fit rough N^2 coefficient using N>=8
    coeffs = []
    for n, t in zip(ns_a, h8):
        if n >= 8:
            coeffs.append(t / (n * n))
    out["hermite8_ms_per_N2"] = float(np.mean(coeffs))

    path = Path(__file__).resolve().parent / "results" / "perf_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nApprox Hermite-8 cost ≈ {out['hermite8_ms_per_N2']:.4f} · N² ms")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
