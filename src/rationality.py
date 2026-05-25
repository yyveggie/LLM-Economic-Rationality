"""Revealed-preference rationality measures.

Implements Afriat's CCEI plus the Houtman-Maks index (HMI), money pump index
(MPI) and minimum cost index (MCI) used in Chen et al. (2023).

Each subject contributes one record:
    decisions: list of (pA, pB, xA, xB)  # 25 budgetary choices
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import itertools
import numpy as np


@dataclass
class Decision:
    pA: float
    pB: float
    xA: float
    xB: float


def _expenditure(d: Decision) -> float:
    return d.pA * d.xA + d.pB * d.xB


def _direct_revealed_pref_matrix(decisions: List[Decision], e: float) -> np.ndarray:
    """A[i, j] = 1 iff bundle i is directly revealed preferred to bundle j at
    cost-efficiency level e. That is, p_i * x_j <= e * p_i * x_i."""
    n = len(decisions)
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        budget_i = e * _expenditure(decisions[i])
        for j in range(n):
            cost_j_at_i = decisions[i].pA * decisions[j].xA + decisions[i].pB * decisions[j].xB
            if cost_j_at_i <= budget_i + 1e-9:
                A[i, j] = True
    return A


def _strict_revealed_pref(decisions: List[Decision], e: float, eps: float = 1e-6) -> np.ndarray:
    """Strict direct revealed preference: p_i * x_j < e * p_i * x_i."""
    n = len(decisions)
    S = np.zeros((n, n), dtype=bool)
    for i in range(n):
        budget_i = e * _expenditure(decisions[i])
        for j in range(n):
            if i == j:
                continue
            cost_j_at_i = decisions[i].pA * decisions[j].xA + decisions[i].pB * decisions[j].xB
            if cost_j_at_i < budget_i - eps:
                S[i, j] = True
    return S


def _violates_garp(decisions: List[Decision], e: float) -> bool:
    """GARP fails iff there exist i, j with x_i R^* x_j and x_j P^0 x_i,
    where R^* is the transitive closure of direct revealed preference R^0.
    """
    A = _direct_revealed_pref_matrix(decisions, e)   # R^0
    S = _strict_revealed_pref(decisions, e)          # P^0
    # Transitive closure via Warshall
    n = A.shape[0]
    R = A.copy()
    for k in range(n):
        R = R | (R[:, k:k + 1] & R[k:k + 1, :])
    # Violation: exists i, j: R[i, j] and S[j, i]
    return bool(np.any(R & S.T))


def ccei(decisions: List[Decision],
         tol: float = 1e-3) -> float:
    """Critical Cost Efficiency Index (Afriat 1972).

    Largest e in [0, 1] for which the data set passes GARP at level e.
    Computed by bisection on a grid.
    """
    if not _violates_garp(decisions, 1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _violates_garp(decisions, mid):
            hi = mid
        else:
            lo = mid
    return lo


def hmi(decisions: List[Decision]) -> int:
    """Houtman-Maks index: minimum number of decisions to remove for the
    remaining set to satisfy GARP. Brute force over subset sizes; n=25 makes
    this expensive but tractable up to ~3 removals; we cap at 5.
    """
    n = len(decisions)
    if not _violates_garp(decisions, 1.0):
        return 0
    for k in range(1, min(n, 5) + 1):
        for combo in itertools.combinations(range(n), k):
            keep = [d for i, d in enumerate(decisions) if i not in combo]
            if not _violates_garp(keep, 1.0):
                return k
    return min(n, 6)  # surrogate when removal count exceeds the cap


def mpi(decisions: List[Decision]) -> float:
    """Money Pump Index (Echenique, Lee & Shum 2011).

    For every observed cycle of length m of strict revealed preference,
    compute (p_i * (x_i - x_j)) summed and normalize by total expenditure.
    We report the mean across detected 2-cycles, a common simplification.
    """
    n = len(decisions)
    A = _direct_revealed_pref_matrix(decisions, 1.0)
    pumps: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j or not (A[i, j] and A[j, i]):
                continue
            cost_i = _expenditure(decisions[i])
            cost_j = _expenditure(decisions[j])
            cost_j_at_i = decisions[i].pA * decisions[j].xA + decisions[i].pB * decisions[j].xB
            cost_i_at_j = decisions[j].pA * decisions[i].xA + decisions[j].pB * decisions[i].xB
            pump = ((cost_i - cost_j_at_i) + (cost_j - cost_i_at_j))
            denom = cost_i + cost_j
            if denom > 0:
                pumps.append(max(0.0, pump / denom))
    return float(np.mean(pumps)) if pumps else 0.0


def mci(decisions: List[Decision]) -> float:
    """Minimum Cost Index (Dean & Martin 2016) approximation.

    The exact MCI requires solving a min-cost edge-removal problem on the
    revealed-preference cycle graph. We use a greedy upper bound: repeatedly
    drop the decision contributing to the most cycles until GARP holds.
    """
    if not _violates_garp(decisions, 1.0):
        return 0.0
    work = list(decisions)
    removed_cost = 0.0
    while _violates_garp(work, 1.0) and len(work) > 2:
        # Score each remaining decision by how many cycles it belongs to.
        n = len(work)
        A = _direct_revealed_pref_matrix(work, 1.0)
        # Symmetric "cycles" approximated as i<->j mutual revealed preference.
        score = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] and A[j, i]:
                    score[i] += 1
                    score[j] += 1
        worst = int(np.argmax(score))
        removed_cost += _expenditure(work[worst])
        work.pop(worst)
    total_cost = sum(_expenditure(d) for d in decisions)
    return float(removed_cost / total_cost) if total_cost > 0 else 0.0


def downward_sloping_spearman(decisions: List[Decision]) -> float:
    """Spearman correlation of ln(xA / xB) and ln(pA / pB).

    A negative value supports the law of demand (downward-sloping demand).
    Implemented with numpy ranks so we do not require scipy at runtime."""
    eps = 1e-6
    log_x = np.log((np.array([d.xA for d in decisions]) + eps) /
                   (np.array([d.xB for d in decisions]) + eps))
    log_p = np.log(np.array([d.pA for d in decisions]) /
                   np.array([d.pB for d in decisions]))
    if np.std(log_x) < 1e-10 or np.std(log_p) < 1e-10:
        return float("nan")
    rx = _rankdata(log_x)
    rp = _rankdata(log_p)
    rx_c = rx - rx.mean()
    rp_c = rp - rp.mean()
    denom = np.sqrt(np.sum(rx_c ** 2) * np.sum(rp_c ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(rx_c * rp_c) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks of a 1-D array, mimicking scipy.stats.rankdata."""
    arr = np.asarray(a, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Average rank for ties
    sorted_arr = arr[order]
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def all_indices(decisions: List[Decision]) -> dict:
    """Compute every rationality measure for one subject's 25 choices."""
    return {
        "ccei": ccei(decisions),
        "hmi": hmi(decisions),
        "mpi": mpi(decisions),
        "mci": mci(decisions),
        "spearman": downward_sloping_spearman(decisions),
    }
