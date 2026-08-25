#!/usr/bin/env python3
"""Export a synthetic 3D Gaussian scene as JSON for the web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _hsv_to_rgb(h: float, s: float, v: float) -> list[float]:
    """h in [0,1), s/v in [0,1] → RGB in [0,1]."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return [float(r), float(g), float(b)]


def _vivid_rgb(rng: np.random.Generator) -> list[float]:
    """Saturated, full-hue color — makes α-blend mistakes visually obvious."""
    return _hsv_to_rgb(
        float(rng.random()),
        float(rng.uniform(0.65, 1.0)),
        float(rng.uniform(0.55, 1.0)),
    )


def main() -> None:
    rng = np.random.default_rng(7)
    n = 1200

    # Three depth clumps (stress early-out / α-blend order)
    mus = []
    scales = []
    colors = []
    opacities = []

    # Background clump (far)
    for _ in range(500):
        mus.append(rng.normal([0, 0, -2.5], [0.85, 0.55, 0.25]))
        scales.append(rng.uniform(0.04, 0.12, 3))
        colors.append(_vivid_rgb(rng))
        opacities.append(rng.uniform(0.15, 0.55))

    # Mid overlapping clump (dense + high opacity)
    for _ in range(400):
        mus.append(rng.normal([0.1, 0.0, 0.2], [0.18, 0.18, 0.13]))
        scales.append(rng.uniform(0.05, 0.14, 3))
        colors.append(_vivid_rgb(rng))
        opacities.append(rng.uniform(0.55, 0.95))

    # Foreground clump
    for _ in range(300):
        mus.append(rng.normal([0.0, 0.1, 1.8], [0.65, 0.42, 0.18]))
        scales.append(rng.uniform(0.03, 0.08, 3))
        colors.append(_vivid_rgb(rng))
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
            "Synthetic scene for T early-out / α-blend experiments. "
            "All clumps use full-hue saturated colors so blend errors show. "
            "opacities are α in [0,1] (classic 3DGS), not optical depth w."
        ),
    }
    path = Path(__file__).with_name("public") / "scene.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"Wrote {path} ({n} gaussians, {path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
