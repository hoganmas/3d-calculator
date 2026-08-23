#!/usr/bin/env python3
"""
Validate analytical 3DGS transmittance / radiance approximations
against numerical quadrature ground truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, norm

PHI = norm.pdf
CDF = norm.cdf


@dataclass
class Gaussians:
    """Depth-ordered mixture components along a ray."""

    mu: np.ndarray  # (N,)
    sigma: np.ndarray  # (N,)
    w: np.ndarray  # (N,) attenuation weights
    c: np.ndarray  # (N,) emission weights

    def sorted(self) -> "Gaussians":
        order = np.argsort(self.mu)
        return Gaussians(
            mu=self.mu[order],
            sigma=self.sigma[order],
            w=self.w[order],
            c=self.c[order],
        )


def density(z: np.ndarray, g: Gaussians) -> np.ndarray:
    """f(z) = sum_i w_i φ((z-μ_i)/σ_i) / σ_i  — PDF-normalized Gaussians.

    Notes use φ((z-μ)/σ) without the 1/σ Jacobian in some places and with
    σ_j outside the Φ₂ integral elsewhere. We treat each component as a
    proper 1D Gaussian density φ_σ(z) = (1/σ) φ((z-μ)/σ), so
      ∫ φ_σ = 1 and Φ_i(z) = CDF((z-μ_i)/σ_i).
    Then ∫_z^{zf} φ_σ = Φ(zf) - Φ(z).
    """
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z, dtype=float)
    for mu, sigma, w in zip(g.mu, g.sigma, g.w):
        out += w * PHI((z - mu) / sigma) / sigma
    return out


def emission(z: np.ndarray, g: Gaussians) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z, dtype=float)
    for mu, sigma, c in zip(g.mu, g.sigma, g.c):
        out += c * PHI((z - mu) / sigma) / sigma
    return out


def optical_depth_exact(z: np.ndarray, zf: float, g: Gaussians) -> np.ndarray:
    """τ(z) = ∫_z^{zf} f(z') dz' = sum_i w_i (Φ_i(zf) - Φ_i(z))."""
    z = np.asarray(z, dtype=float)
    tau = np.zeros_like(z, dtype=float)
    for mu, sigma, w in zip(g.mu, g.sigma, g.w):
        tau += w * (CDF((zf - mu) / sigma) - CDF((z - mu) / sigma))
    return tau


def T_exact(z: np.ndarray, zf: float, g: Gaussians) -> np.ndarray:
    return np.exp(-optical_depth_exact(z, zf, g))


def appendix_a_params(
    w: float | np.ndarray, sigma: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Appendix A: mean shift δ and scale ν for the CDF match.

    exp(w Φ((z-μ)/σ)) ≈ 1 + (e^w - 1) Φ((z - μ - δ)/ν)

    δ = (σ/w)(-√(2π) + √(2π + 2 w²))   [inflection]
    ν = σ |e^w-1| exp(-w Φ(δ/σ)) / (w φ(δ/σ) √(2π))   [slope at δ]

    As w → 0: δ → σ w / √(2π), ν → σ.
    """
    w_b, s_b = np.broadcast_arrays(
        np.asarray(w, dtype=float), np.asarray(sigma, dtype=float)
    )
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    small = np.abs(w_b) < 1e-8

    delta = np.where(
        small,
        s_b * w_b / sqrt_2pi,
        (s_b / w_b) * (-sqrt_2pi + np.sqrt(2.0 * np.pi + 2.0 * w_b**2)),
    )

    # ν from matching f'(δ); use limiting ν=σ for tiny w
    u = delta / s_b
    # Avoid underflow/overflow in exp(w)*CDF path via expm1 and clip of pdf
    phi_u = PHI(u)
    # Guard: phi_u can be tiny for large |u|, but for our δ/σ ∈ (0, √2) it's fine
    nu = np.where(
        small,
        s_b,
        s_b
        * np.abs(np.expm1(w_b))
        * np.exp(-w_b * CDF(u))
        / (w_b * phi_u * sqrt_2pi),
    )
    return delta, nu


def appendix_a_mean_shift(w: float | np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    """Backward-compatible: inflection mean shift only."""
    delta, _ = appendix_a_params(w, sigma)
    return delta


def exp_w_phi_approx(z: np.ndarray, mu: float, sigma: float, w: float) -> np.ndarray:
    """Appendix A: inflection + slope matched shifted CDF."""
    delta, nu = appendix_a_params(w, sigma)
    return 1.0 + (np.exp(w) - 1.0) * CDF((z - mu - float(delta)) / float(nu))


def T_approx(z: np.ndarray, zf: float, g: Gaussians) -> np.ndarray:
    """
    T(z) ≈ A (1 + sum_i B_i Φ(h_i(z)))
    with depth-ordered gaussians, product-of-CDFs collapsed to farthest mean.
    h_i(z) = (z - μ_i - δ_i) / ν_i
    """
    g = g.sorted()
    z = np.asarray(z, dtype=float)

    A = np.exp(-np.sum(g.w * CDF((zf - g.mu) / g.sigma)))
    # B_i = (e^{w_i} - 1) * prod_{j<i} e^{w_j}
    log_prefix = np.cumsum(g.w) - g.w  # sum of w_j for j < i
    B = (np.exp(g.w) - 1.0) * np.exp(log_prefix)

    delta, nu = appendix_a_params(g.w, g.sigma)
    h = (z[..., None] - g.mu - delta) / nu
    return A * (1.0 + np.sum(B * CDF(h), axis=-1))


def phi_times_cdf_integral(
    zi: float, zf: float, mu_j: float, sigma_j: float, mu_i: float, sigma_i: float
) -> float:
    """
    ∫_{zi}^{zf} φ_σj(z) Φ((z - μ_i)/σ_i) dz
      = Φ2(a, u_f; ρ) - Φ2(a, u_i; ρ)
    with a = (μ_j - μ_i) / sqrt(σ_i² + σ_j²),
         ρ = -σ_j / sqrt(σ_i² + σ_j²),
         u = (z - μ_j)/σ_j.

    Derived from ∫ φ(u) Φ((σ_j u + μ_j - μ_i)/σ_i) du after u=(z-μ_j)/σ_j,
    and the density already includes 1/σ_j so the σ_j from du cancels.
    """
    s = np.sqrt(sigma_i**2 + sigma_j**2)
    a = (mu_j - mu_i) / s
    rho = -sigma_j / s
    cov = [[1.0, rho], [rho, 1.0]]
    mean = [0.0, 0.0]

    def bivariate_cdf(u: float) -> float:
        # P(X <= a, Y <= u)
        return float(multivariate_normal.cdf([a, u], mean=mean, cov=cov))

    ui = (zi - mu_j) / sigma_j
    uf = (zf - mu_j) / sigma_j
    return bivariate_cdf(uf) - bivariate_cdf(ui)


def radiance_approx(zi: float, zf: float, g: Gaussians) -> float:
    """
    ∫ g(z) T(z) dz ≈ A [ ∫ g + sum_i B_i ∫ g(z) Φ(h_i(z)) dz ]

    Note: the writeup writes A(1 + sum ...), which drops ∫g for the constant
    term. We use the correct expansion A(∫g + sum B_i ∫g Φ(h_i)).
    Φ(h_i) = Φ((z - μ_i - δ_i)/ν_i) uses Appendix A mean+scale.
    """
    g = g.sorted()
    A = float(np.exp(-np.sum(g.w * CDF((zf - g.mu) / g.sigma))))
    log_prefix = np.cumsum(g.w) - g.w
    B = (np.exp(g.w) - 1.0) * np.exp(log_prefix)

    # ∫_{zi}^{zf} g(z) dz
    integ_g = 0.0
    for mu, sigma, c in zip(g.mu, g.sigma, g.c):
        integ_g += c * (CDF((zf - mu) / sigma) - CDF((zi - mu) / sigma))

    # sum_i sum_j B_i c_j ∫ φ_j Φ((z - μ_i - δ_i)/ν_i)
    deltas, nus = appendix_a_params(g.w, g.sigma)
    cross = 0.0
    for i in range(len(g.mu)):
        mu_i_shift = float(g.mu[i] + deltas[i])
        nu_i = float(nus[i])
        for j in range(len(g.mu)):
            cross += (
                B[i]
                * g.c[j]
                * phi_times_cdf_integral(
                    zi, zf, g.mu[j], g.sigma[j], mu_i_shift, nu_i
                )
            )

    return A * (integ_g + cross)


def radiance_exact(zi: float, zf: float, g: Gaussians, n: int = 8192) -> float:
    """Trapezoidal / midpoint quadrature of g(z) T(z)."""
    z = np.linspace(zi, zf, n)
    integrand = emission(z, g) * T_exact(z, zf, g)
    return float(np.trapezoid(integrand, z))


def appendix_a_error(w: float, sigma: float = 1.0, n: int = 2001) -> dict:
    """Pointwise error for exp(w Φ) ≈ shifted CDF form, over ±8σ."""
    x = np.linspace(-8 * sigma, 8 * sigma, n)
    exact = np.exp(w * CDF(x / sigma))
    approx = exp_w_phi_approx(x, 0.0, sigma, w)
    # Previous: mean-matched only (ν = σ)
    delta_mean_only = float(appendix_a_mean_shift(w, sigma))
    approx_mean_only = 1.0 + (np.exp(w) - 1.0) * CDF((x - delta_mean_only) / sigma)
    # Original incorrect: δ = w σ², ν = σ
    approx_v0 = 1.0 + (np.exp(w) - 1.0) * CDF((x - w * sigma**2) / sigma)
    err = approx - exact
    err_mean = approx_mean_only - exact
    err_v0 = approx_v0 - exact
    delta, nu = appendix_a_params(w, sigma)
    return {
        "w": w,
        "delta": float(delta),
        "nu": float(nu),
        "nu_over_sigma": float(nu) / sigma,
        "max_abs": float(np.max(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rel_l2": float(np.linalg.norm(err) / (np.linalg.norm(exact) + 1e-15)),
        "rel_l2_mean_only": float(
            np.linalg.norm(err_mean) / (np.linalg.norm(exact) + 1e-15)
        ),
        "rel_l2_v0": float(np.linalg.norm(err_v0) / (np.linalg.norm(exact) + 1e-15)),
    }


def T_error(zi: float, zf: float, g: Gaussians, n: int = 2001) -> dict:
    z = np.linspace(zi, zf, n)
    exact = T_exact(z, zf, g)
    approx = T_approx(z, zf, g)
    err = approx - exact
    return {
        "max_abs": float(np.max(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rel_l2": float(np.linalg.norm(err) / (np.linalg.norm(exact) + 1e-15)),
        "endpoint_zi": {"exact": float(exact[0]), "approx": float(approx[0])},
        "endpoint_zf": {"exact": float(exact[-1]), "approx": float(approx[-1])},
    }


def sample_curve(zi: float, zf: float, g: Gaussians, n: int = 400) -> dict:
    z = np.linspace(zi, zf, n)
    return {
        "z": z.tolist(),
        "T_exact": T_exact(z, zf, g).tolist(),
        "T_approx": T_approx(z, zf, g).tolist(),
        "f": density(z, g).tolist(),
        "g": emission(z, g).tolist(),
    }


def make_scene(
    name: str,
    mus: list[float],
    sigmas: list[float],
    ws: list[float],
    cs: list[float] | None = None,
) -> tuple[str, Gaussians]:
    mu = np.asarray(mus, dtype=float)
    sigma = np.asarray(sigmas, dtype=float)
    w = np.asarray(ws, dtype=float)
    c = np.asarray(cs if cs is not None else ws, dtype=float)
    return name, Gaussians(mu=mu, sigma=sigma, w=w, c=c)


def main() -> None:
    zi, zf = -4.0, 8.0

    scenes = [
        make_scene("single_weak", [2.0], [0.5], [0.3]),
        make_scene("single_moderate", [2.0], [0.5], [1.0]),
        make_scene("single_strong", [2.0], [0.5], [3.0]),
        make_scene(
            "two_separated",
            [1.0, 5.0],
            [0.4, 0.4],
            [1.0, 1.0],
        ),
        make_scene(
            "two_adjacent",
            [2.0, 3.0],
            [0.4, 0.4],
            [1.0, 1.0],
        ),
        make_scene(
            "two_heavy_overlap",
            [2.0, 2.3],
            [0.6, 0.6],
            [1.2, 1.2],
        ),
        make_scene(
            "three_separated",
            [0.5, 3.0, 6.0],
            [0.35, 0.35, 0.35],
            [0.8, 0.8, 0.8],
        ),
        make_scene(
            "three_clustered",
            [2.0, 2.5, 3.0],
            [0.45, 0.45, 0.45],
            [1.0, 1.0, 1.0],
        ),
        make_scene(
            "unequal_weights",
            [1.0, 3.0, 5.5],
            [0.4, 0.5, 0.3],
            [0.4, 2.5, 0.7],
            [1.0, 1.0, 1.0],
        ),
    ]

    appendix_a = [appendix_a_error(w) for w in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]]

    scene_results = []
    curves = {}
    for name, g in scenes:
        te = T_error(zi, zf, g)
        r_ex = radiance_exact(zi, zf, g)
        r_ap = radiance_approx(zi, zf, g)
        scene_results.append(
            {
                "name": name,
                "N": int(len(g.mu)),
                "T": te,
                "radiance": {
                    "exact": r_ex,
                    "approx": r_ap,
                    "abs_err": abs(r_ap - r_ex),
                    "rel_err": abs(r_ap - r_ex) / (abs(r_ex) + 1e-15),
                },
            }
        )
        curves[name] = sample_curve(zi, zf, g)

    # Weight sweep for two separated Gaussians (product approx should be good)
    weight_sweep = []
    for w in np.linspace(0.1, 3.0, 12):
        _, g = make_scene("sweep", [1.0, 5.0], [0.4, 0.4], [float(w), float(w)])
        te = T_error(zi, zf, g)
        r_ex = radiance_exact(zi, zf, g)
        r_ap = radiance_approx(zi, zf, g)
        weight_sweep.append(
            {
                "w": float(w),
                "T_rel_l2": te["rel_l2"],
                "T_max_abs": te["max_abs"],
                "rad_rel_err": abs(r_ap - r_ex) / (abs(r_ex) + 1e-15),
            }
        )

    # Separation sweep: fix w=1, vary Δμ / σ
    sep_sweep = []
    for dmu in np.linspace(0.1, 4.0, 16):
        _, g = make_scene("sep", [2.0, 2.0 + float(dmu)], [0.5, 0.5], [1.0, 1.0])
        te = T_error(zi, zf, g)
        r_ex = radiance_exact(zi, zf, g)
        r_ap = radiance_approx(zi, zf, g)
        sep_sweep.append(
            {
                "dmu_over_sigma": float(dmu / 0.5),
                "dmu": float(dmu),
                "T_rel_l2": te["rel_l2"],
                "T_max_abs": te["max_abs"],
                "rad_rel_err": abs(r_ap - r_ex) / (abs(r_ex) + 1e-15),
            }
        )

    out = {
        "interval": {"zi": zi, "zf": zf},
        "appendix_a": appendix_a,
        "scenes": scene_results,
        "weight_sweep": weight_sweep,
        "separation_sweep": sep_sweep,
        "curves": {
            k: curves[k]
            for k in [
                "single_moderate",
                "two_separated",
                "two_heavy_overlap",
                "three_clustered",
            ]
        },
        "notes": {
            "radiance_constant_term": (
                "Used A(∫g + Σ B_i ∫g Φ(h_i)), not A(1 + Σ ...). "
                "The bare '1' in the notes is incorrect for the radiance integral."
            ),
            "gaussian_normalization": (
                "Components treated as w * Normal(μ, σ²) densities "
                "(include 1/σ), matching Φ_i(z) = CDF((z-μ)/σ)."
            ),
        },
    }

    out_path = Path(__file__).with_name("validation_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    print("\n=== Appendix A (mean+scale vs mean-only vs v0) ===")
    for row in appendix_a:
        print(
            f"  w={row['w']:.1f}  δ={row['delta']:.3f}  ν/σ={row['nu_over_sigma']:.3f}  "
            f"rel_L2={row['rel_l2']:.3e}  "
            f"(mean-only {row['rel_l2_mean_only']:.3e}, v0 {row['rel_l2_v0']:.3e})"
        )

    print("\n=== Scene transmittance / radiance ===")
    for s in scene_results:
        print(
            f"  {s['name']:20s}  T_rel_L2={s['T']['rel_l2']:.3e}  "
            f"rad_rel={s['radiance']['rel_err']:.3e}  "
            f"rad_ex={s['radiance']['exact']:.4f}  rad_ap={s['radiance']['approx']:.4f}"
        )


if __name__ == "__main__":
    main()
