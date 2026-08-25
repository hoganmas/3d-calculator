#!/usr/bin/env python3
"""Export a synthetic 3D Gaussian scene as JSON for the web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(7)
    n = 1200

    # Two depth layers + a dense overlapping clump (stress early-out / order)
    mus = []
    scales = []
    colors = []
    opacities = []

    # Background layer (far)
    for _ in range(500):
        mus.append(rng.normal([0, 0, -2.5], [1.2, 0.8, 0.35]))
        scales.append(rng.uniform(0.04, 0.12, 3))
        colors.append(rng.uniform([0.2, 0.35, 0.7], [0.55, 0.7, 1.0]))
        opacities.append(rng.uniform(0.15, 0.55))

    # Mid overlapping clump (dense + high opacity → early-out stress)
    for _ in range(400):
        mus.append(rng.normal([0.1, 0.0, 0.2], [0.25, 0.25, 0.18]))
        scales.append(rng.uniform(0.05, 0.14, 3))
        colors.append(rng.uniform([0.75, 0.25, 0.15], [1.0, 0.55, 0.35]))
        opacities.append(rng.uniform(0.55, 0.95))

    # Foreground sparse bright
    for _ in range(300):
        mus.append(rng.normal([0.0, 0.1, 1.8], [0.9, 0.6, 0.25]))
        scales.append(rng.uniform(0.03, 0.08, 3))
        colors.append(rng.uniform([0.85, 0.85, 0.2], [1.0, 1.0, 0.55]))
        opacities.append(rng.uniform(0.2, 0.7))

    mus = np.asarray(mus[:n], dtype=np.float32)
    scales = np.asarray(scales[:n], dtype=np.float32)
    colors = np.asarray(colors[:n], dtype=np.float32)
    opacities = np.asarray(opacities[:n], dtype=np.float32)

    # Unit quaternions (w,x,y,z) — identity + small jitter
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    jitter = rng.normal(0, 0.05, (n, 3)).astype(np.float32)
    quats[:, 1:] = jitter
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    out = {
        "format": "web-gsplat-v1",
        "count": n,
        "means": mus.tolist(),
        "scales": scales.tolist(),
        "quats": quats.tolist(),
        "colors": colors.tolist(),
        "opacities": opacities.tolist(),
        "notes": (
            "Synthetic scene for T early-out experiments. "
            "opacities are α in [0,1] (classic 3DGS), not optical depth w."
        ),
    }
    path = Path(__file__).with_name("public") / "scene.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"Wrote {path} ({n} gaussians, {path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
