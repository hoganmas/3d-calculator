#!/usr/bin/env python3
"""
Stress / sweep tests for the polynomial transmittance approximation.

Focuses on:
  - degree and attenuation strength sweeps
  - Appendix-A (product) vs product-collapse error split
  - optional curve dumps for plotting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validate_poly import (
    PolyScene,
    T_collapsed_approx,
    T_exact,
    T_product_approx,
    build_T_approx,
    evaluate_scene,
    err_stats,
)


def sweep_linear_strength(strengths: list[float], zf: float = 1.0) -> list[dict]:
    rows = []
    for a in strengths:
        scene = PolyScene(0.0, zf, w=np.array([a]), c=np.array([1.0]))
        r = evaluate_scene(scene)
        rows.append(
            {
                "A_w0": a,
                "T_product_rel_l2": r["T"]["product_vs_exact"]["rel_l2"],
                "T_collapse_rel_l2": r["T"]["collapsed_vs_exact"]["rel_l2"],
                "rad_rel": r["radiance"]["rel_err_analytic"],
                "factor_case": r["factors"][0]["case"],
            }
        )
    return rows


def sweep_degree(max_deg: int = 5, scale: float = 0.4) -> list[dict]:
    rows = []
    for d in range(0, max_deg + 1):
        w = scale * np.ones(d + 1)
        scene = PolyScene(0.0, 1.0, w=w, c=np.array([1.0, 0.2]))
        r = evaluate_scene(scene)
        rows.append(
            {
                "deg_f": d,
                "T_product_rel_l2": r["T"]["product_vs_exact"]["rel_l2"],
                "T_collapse_rel_l2": r["T"]["collapsed_vs_exact"]["rel_l2"],
                "collapse_vs_product_max": r["T"]["collapse_vs_product"]["max_abs"],
                "rad_rel": r["radiance"]["rel_err_analytic"],
            }
        )
    return rows


def dump_curves(scene: PolyScene, path: Path, n: int = 500) -> None:
    parts = build_T_approx(scene)
    z = np.linspace(scene.z0, scene.zf, n)
    exact = T_exact(z, scene)
    product = T_product_approx(z, parts)
    collapsed = T_collapsed_approx(z, parts)
    np.savez(
        path,
        z=z,
        T_exact=exact,
        T_product=product,
        T_collapsed=collapsed,
        err_product=product - exact,
        err_collapsed=collapsed - exact,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("poly_stress_results.json"))
    ap.add_argument("--curves-dir", type=Path, default=None)
    args = ap.parse_args()

    strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    payload = {
        "linear_strength_sweep": sweep_linear_strength(strengths),
        "degree_sweep": sweep_degree(),
    }

    # A few multi-term random draws for collapse pressure
    rng = np.random.default_rng(1)
    rand_rows = []
    for k in range(15):
        d = int(rng.integers(1, 5))
        w = rng.uniform(0.05, 1.0, size=d + 1)
        scene = PolyScene(0.0, float(rng.uniform(0.8, 1.5)), w=w, c=np.array([1.0]))
        r = evaluate_scene(scene)
        rand_rows.append(
            {
                "k": k,
                "deg": d,
                "w": w.tolist(),
                "T_product_rel_l2": r["T"]["product_vs_exact"]["rel_l2"],
                "T_collapse_rel_l2": r["T"]["collapsed_vs_exact"]["rel_l2"],
                "collapse_vs_product_max": r["T"]["collapse_vs_product"]["max_abs"],
            }
        )
    payload["random_multiterm"] = rand_rows

    print("Linear strength sweep (w=[A], g=1):")
    for row in payload["linear_strength_sweep"]:
        print(
            f"  A={row['A_w0']:<5}  product={row['T_product_rel_l2']:.3e}  "
            f"collapse={row['T_collapse_rel_l2']:.3e}  rad={row['rad_rel']:.3e}"
        )

    print("\nDegree sweep (w_i=0.4):")
    for row in payload["degree_sweep"]:
        print(
            f"  deg={row['deg_f']}  product={row['T_product_rel_l2']:.3e}  "
            f"collapse={row['T_collapse_rel_l2']:.3e}  "
            f"col-prod max={row['collapse_vs_product_max']:.3e}"
        )

    if args.curves_dir is not None:
        args.curves_dir.mkdir(parents=True, exist_ok=True)
        demo = {
            "linear": PolyScene(0.0, 1.0, w=np.array([2.0]), c=np.array([1.0])),
            "quadratic": PolyScene(0.0, 1.0, w=np.array([0.5, 0.0, 2.0]), c=np.array([1.0])),
            "cubic": PolyScene(0.0, 1.0, w=np.array([0.3, 0.2, 0.1, 0.5]), c=np.array([1.0])),
        }
        for name, scene in demo.items():
            dump_curves(scene, args.curves_dir / f"{name}.npz")
            # also print pointwise stats
            parts = build_T_approx(scene)
            z = np.linspace(scene.z0, scene.zf, 2001)
            print(
                f"\ncurves {name}:",
                err_stats(T_collapsed_approx(z, parts), T_exact(z, scene)),
            )

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
