"""Structural estimation of preference parameters (alpha, rho).

Risk and Time domains use the disappointment-aversion / time-separable form:
    U(x1, x2) = alpha * u(x1) + (1 - alpha) * u(x2)
    u(z)     = z^rho / rho      (rho != 0)
              ln(z)             (rho = 0)

Social and Food domains use a CES utility:
    U(x1, x2) = (alpha * x1^rho + (1 - alpha) * x2^rho)^(1/rho)

We fit by NLLS on the log demand-share equation:
    ln(x1 / x2) = (1 / (rho - 1)) * ln(p1 / p2) + ln((1 - alpha) / alpha)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import least_squares

from .rationality import Decision


def _prepare(decisions: List[Decision], domain: str) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-3
    pA = np.array([d.pA for d in decisions])
    pB = np.array([d.pB for d in decisions])
    xA = np.array([d.xA for d in decisions])
    xB = np.array([d.xB for d in decisions])

    if domain == "risk":
        # x1 = max{xA, xB} (better outcome)
        x1 = np.maximum(xA, xB)
        x2 = np.minimum(xA, xB)
        # The corresponding price is the lower price for the better outcome
        # Because the agent shifts to the cheaper one to magnify x1.
        p1 = np.where(xA >= xB, pA, pB)
        p2 = np.where(xA >= xB, pB, pA)
    else:  # time / social / food
        x1, x2 = xA, xB
        p1, p2 = pA, pB

    return (np.log((p1 + eps) / (p2 + eps)),
            np.log((x1 + eps) / (x2 + eps)))


def estimate(decisions: List[Decision], domain: str) -> dict:
    """Returns {alpha, rho, alpha_se, rho_se} or NaNs if estimation fails."""
    log_p, log_x = _prepare(decisions, domain)

    def residuals(theta: np.ndarray) -> np.ndarray:
        alpha, rho = theta
        if not (0.0 < alpha < 1.0) or rho >= 1.0:
            return np.full_like(log_x, 1e3)
        slope = 1.0 / (rho - 1.0)
        intercept = np.log((1.0 - alpha) / alpha)
        return log_x - (slope * log_p + intercept)

    try:
        sol = least_squares(
            residuals,
            x0=np.array([0.5, -0.5]),
            bounds=([1e-4, -10.0], [1.0 - 1e-4, 0.999]),
            max_nfev=400,
        )
        alpha, rho = float(sol.x[0]), float(sol.x[1])
        # Approximate standard errors via the Jacobian
        J = sol.jac
        n_obs = len(log_x)
        if n_obs > 2 and J.shape[1] == 2:
            sigma2 = float(np.sum(sol.fun ** 2) / max(1, n_obs - 2))
            try:
                cov = sigma2 * np.linalg.inv(J.T @ J)
                alpha_se = float(np.sqrt(max(cov[0, 0], 0)))
                rho_se = float(np.sqrt(max(cov[1, 1], 0)))
            except np.linalg.LinAlgError:
                alpha_se = rho_se = float("nan")
        else:
            alpha_se = rho_se = float("nan")
        return {"alpha": alpha, "rho": rho,
                "alpha_se": alpha_se, "rho_se": rho_se}
    except Exception:
        return {"alpha": float("nan"), "rho": float("nan"),
                "alpha_se": float("nan"), "rho_se": float("nan")}
