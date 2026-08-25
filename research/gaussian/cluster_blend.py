#!/usr/bin/env python3
"""
Sketch: overlap-adaptive cluster compositing for Gaussian RTE.

Pipeline
--------
1. Sort Gaussians by depth (μ ascending = away from camera at zf).
2. Merge into overlap clusters when consecutive means satisfy
     |μ_{i+1} - μ_i| <= k (σ_i + σ_{i+1}).
3. Front-to-back composite clusters (camera at zf):
     C += T_front * I_cluster
     T_front *= exp(-Σ_{i∈cluster} w_i)   # full-mass exit transmittance

Cluster integral I_cluster = ∫ g_cluster(z) T_cluster(z) dz, where T_cluster
uses *only* the cluster's extinction (front attenuation factored into T_front).

Integrators (swap inside a cluster)
---------------------------------
  hermite   — exact T_cluster + Gauss-Hermite  (oracle)
  analytic  — Appendix A + collapse + Φ₂ on the cluster only
  alpha     — classic albedo·α  (baseline)

For clusters with N>max_analytic_n, analytic path falls back to hermite
(so the "novel cell" is specialized to small-N, especially pairs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blend_benchmark import (
    classic_alpha_blend,
    exact_T_hermite,
    rel_err,
)
from validate_approx import (
    Gaussians,
    make_scene,
    radiance_approx,
    radiance_exact,
)


@dataclass
class ClusterStats:
    n_clusters: int
    sizes: list[int]
    n_singleton: int
    n_pair: int
    n_large: int


def cluster_indices(g: Gaussians, k_sigma: float = 2.5) -> list[list[int]]:
    """Greedy merge of depth-sorted Gaussians into overlap clusters."""
    g = g.sorted()
    n = len(g.mu)
    if n == 0:
        return []
    clusters: list[list[int]] = [[0]]
    for i in range(1, n):
        prev = clusters[-1][-1]
        gap = g.mu[i] - g.mu[prev]
        thresh = k_sigma * (g.sigma[i] + g.sigma[prev])
        if gap <= thresh:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def subset(g: Gaussians, idxs: list[int]) -> Gaussians:
    ix = np.asarray(idxs, dtype=int)
    return Gaussians(
        mu=g.mu[ix],
        sigma=g.sigma[ix],
        w=g.w[ix],
        c=g.c[ix],
    )


def cluster_stats(clusters: list[list[int]]) -> ClusterStats:
    sizes = [len(c) for c in clusters]
    return ClusterStats(
        n_clusters=len(clusters),
        sizes=sizes,
        n_singleton=sum(1 for s in sizes if s == 1),
        n_pair=sum(1 for s in sizes if s == 2),
        n_large=sum(1 for s in sizes if s > 2),
    )


def cluster_interval(cl: Gaussians, zi: float, zf: float, pad: float = 4.0) -> tuple[float, float]:
    """Local integration window covering the cluster (±pad σ)."""
    lo = float(np.min(cl.mu - pad * cl.sigma))
    hi = float(np.max(cl.mu + pad * cl.sigma))
    return max(zi, lo), min(zf, hi)


def integrate_cluster(
    cl: Gaussians,
    zi: float,
    zf: float,
    method: str,
    *,
    hermite_nodes: int = 8,
) -> float:
    """I = ∫ g_cl T_cl over [zi, zf] with cluster-only extinction."""
    if len(cl.mu) == 0:
        return 0.0
    z0, z1 = cluster_interval(cl, zi, zf)
    if z1 <= z0:
        return 0.0

    if method == "hermite":
        return exact_T_hermite(cl, z0, z1, n_nodes=hermite_nodes)
    if method == "analytic":
        return radiance_approx(z0, z1, cl)
    if method == "alpha":
        return classic_alpha_blend(cl, z0, z1)
    if method == "trap":
        return radiance_exact(z0, z1, cl)
    raise ValueError(method)


def cluster_composite(
    g: Gaussians,
    zi: float,
    zf: float,
    *,
    method: str = "hermite",
    k_sigma: float = 2.5,
    max_analytic_n: int = 2,
    hermite_nodes: int = 8,
) -> tuple[float, ClusterStats, dict]:
    """
    Front-to-back cluster compositing.

    method selects the *preferred* cluster integrator. For method='analytic',
    clusters larger than max_analytic_n fall back to hermite (pair/specialized cell).
    """
    g = g.sorted()
    clusters = cluster_indices(g, k_sigma=k_sigma)
    stats = cluster_stats(clusters)

    # Front-to-back: reverse cluster order (largest μ first)
    T_front = 1.0
    C = 0.0
    detail = {"per_cluster": []}

    for idxs in reversed(clusters):
        cl = subset(g, idxs)
        local = method
        if method == "analytic" and len(idxs) > max_analytic_n:
            local = "hermite"

        I_cl = integrate_cluster(
            cl, zi, zf, local, hermite_nodes=hermite_nodes
        )
        contrib = T_front * I_cl
        C += contrib

        # Exit transmittance of this cluster (full optical mass)
        tau_cl = float(np.sum(cl.w))
        T_exit = float(np.exp(-tau_cl))
        detail["per_cluster"].append(
            {
                "size": len(idxs),
                "integrator": local,
                "I_cl": I_cl,
                "T_front": T_front,
                "contrib": contrib,
                "T_exit": T_exit,
            }
        )
        T_front *= T_exit
        if T_front < 1e-6:
            break

    return float(C), stats, detail


def evaluate_sketch(
    name: str,
    g: Gaussians,
    zi: float,
    zf: float,
    k_sigma: float = 2.5,
) -> dict:
    g = g.sorted()
    gt = radiance_exact(zi, zf, g)

    global_h = exact_T_hermite(g, zi, zf)
    global_a = radiance_approx(zi, zf, g)
    global_alpha = classic_alpha_blend(g, zi, zf)

    cl_h, st, _ = cluster_composite(g, zi, zf, method="hermite", k_sigma=k_sigma)
    cl_a, _, det_a = cluster_composite(
        g, zi, zf, method="analytic", k_sigma=k_sigma, max_analytic_n=2
    )
    cl_alpha, _, _ = cluster_composite(g, zi, zf, method="alpha", k_sigma=k_sigma)

    # Analytic on pairs only, hermite fallback — also try max_analytic_n=3
    cl_a3, _, _ = cluster_composite(
        g, zi, zf, method="analytic", k_sigma=k_sigma, max_analytic_n=3
    )

    return {
        "name": name,
        "N": int(len(g.mu)),
        "gt": gt,
        "clusters": {
            "n": st.n_clusters,
            "sizes": st.sizes,
            "singleton": st.n_singleton,
            "pair": st.n_pair,
            "large": st.n_large,
        },
        "rel_err": {
            "global_hermite": rel_err(global_h, gt),
            "global_analytic": rel_err(global_a, gt),
            "global_alpha": rel_err(global_alpha, gt),
            "cluster_hermite": rel_err(cl_h, gt),
            "cluster_analytic_n2": rel_err(cl_a, gt),
            "cluster_analytic_n3": rel_err(cl_a3, gt),
            "cluster_alpha": rel_err(cl_alpha, gt),
        },
        "values": {
            "global_hermite": global_h,
            "cluster_hermite": cl_h,
            "cluster_analytic_n2": cl_a,
            "cluster_analytic_n3": cl_a3,
        },
        "analytic_pair_detail": det_a["per_cluster"],
    }


def main() -> None:
    zi, zf = -4.0, 8.0
    scenes: list[tuple[str, Gaussians]] = [
        make_scene("two_separated", [1.0, 5.0], [0.4, 0.4], [1.0, 1.0]),
        make_scene("two_heavy_overlap", [2.0, 2.3], [0.6, 0.6], [1.2, 1.2]),
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
        make_scene(
            "three_clustered",
            [2.0, 2.5, 3.0],
            [0.45, 0.45, 0.45],
            [1.0, 1.0, 1.0],
        ),
        (
            "three_overlap_diff_alb",
            Gaussians(
                mu=np.array([2.0, 2.4, 2.8]),
                sigma=np.array([0.4, 0.4, 0.4]),
                w=np.array([0.8, 1.0, 0.6]),
                c=np.array([0.3, 1.5, 0.9]),
            ),
        ),
        # Mixed: two pairs separated
        (
            "two_pairs",
            Gaussians(
                mu=np.array([1.0, 1.3, 6.0, 6.4]),
                sigma=np.array([0.35, 0.35, 0.4, 0.4]),
                w=np.array([0.7, 0.9, 1.0, 0.5]),
                c=np.array([0.2, 1.4, 0.4, 1.8]),
            ),
        ),
        # Crowded cluster of 5
        (
            "five_crowd_diff_alb",
            Gaussians(
                mu=np.linspace(2.0, 3.2, 5),
                sigma=np.full(5, 0.35),
                w=np.array([0.5, 0.8, 1.2, 0.6, 0.9]),
                c=np.array([0.2, 1.5, 0.4, 1.8, 0.7]),
            ),
        ),
    ]

    # Also a longer ray with many separated + one overlap pair
    mus = list(np.linspace(0.5, 9.0, 8))
    mus[3] = mus[2] + 0.25  # force one overlap pair
    scenes.append(
        (
            "mostly_sep_one_pair",
            Gaussians(
                mu=np.array(mus),
                sigma=np.full(8, 0.25),
                w=np.full(8, 0.5),
                c=np.array([0.3, 1.0, 0.5, 1.5, 0.8, 0.2, 1.2, 0.6]),
            ),
        )
    )

    rows = []
    print(
        f"{'scene':24s}  {'clu':9s}  {'gH':8s}  {'cH':8s}  {'cA2':8s}  {'cA3':8s}  {'α':8s}  {'gA':8s}"
    )
    for name, g in scenes:
        row = evaluate_sketch(name, g, zi, zf, k_sigma=2.5)
        rows.append(row)
        cl = row["clusters"]
        e = row["rel_err"]
        print(
            f"{name:24s}  "
            f"{cl['n']}/{cl['pair']}p/{cl['large']}L  "
            f"{e['global_hermite']:.2e}  "
            f"{e['cluster_hermite']:.2e}  "
            f"{e['cluster_analytic_n2']:.2e}  "
            f"{e['cluster_analytic_n3']:.2e}  "
            f"{e['global_alpha']:.2e}  "
            f"{e['global_analytic']:.2e}"
        )

    # k_sigma sweep on the hard case
    print("\n=== k_sigma sweep on five_crowd_diff_alb ===")
    _, g_hard = scenes[-2]
    sweeps = []
    for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        row = evaluate_sketch("five_crowd", g_hard, zi, zf, k_sigma=k)
        sweeps.append(
            {
                "k_sigma": k,
                "clusters": row["clusters"],
                "rel_err": row["rel_err"],
            }
        )
        cl = row["clusters"]
        e = row["rel_err"]
        print(
            f"  k={k:.1f}  sizes={cl['sizes']}  "
            f"cH={e['cluster_hermite']:.3e}  "
            f"cA2={e['cluster_analytic_n2']:.3e}  "
            f"cA3={e['cluster_analytic_n3']:.3e}"
        )

    out = {
        "interval": {"zi": zi, "zf": zf},
        "scenes": rows,
        "k_sigma_sweep": sweeps,
        "sketch": {
            "idea": (
                "Overlap clusters + T_front compositing; "
                "Hermite oracle inside clusters; "
                "Appendix-A analytic as drop-in for N<=2 (or 3) cells."
            ),
            "next": [
                "Replace pair analytic with exact-T Hermite residual fit / better ΦiΦj",
                "GPU: tile-local clustering on sorted hit lists",
                "Train with cluster-Hermite teacher, analytic student",
            ],
        },
    }
    path = Path(__file__).resolve().parent / "results" / "cluster_sketch_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
