"""Parse a model's free-form reply into the (xA, xB) allocation."""
from __future__ import annotations

import re
from typing import Optional, Tuple
from .tasks import BudgetTask


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _all_numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text)]


def parse_continuous(text: str, total: float = 100.0,
                     tol: float = 1.0) -> Optional[Tuple[float, float]]:
    """Parse "xA points to A, xB points to B" replies.

    Strategy: find the first pair of numbers in [0, total] that sums to ~total.
    Falls back to (n, total - n) if only a single number is present.
    """
    nums = _all_numbers(text)
    nums = [n for n in nums if 0.0 <= n <= total]
    # Look for a pair (a, b) with a + b ≈ total
    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        if abs(a + b - total) <= tol:
            return a, b
    # Try non-adjacent pairs
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            if abs(a + b - total) <= tol:
                return a, b
    # Single-number fallback (e.g. "I will invest 60 in A")
    if nums:
        a = nums[0]
        if 0.0 <= a <= total:
            return a, total - a
    return None


def parse_discrete(text: str, num_options: int = 11) -> Optional[int]:
    """Return the chosen option index (0..num_options-1)."""
    # Match "Option 3", "option #3", "(M3, N3)", "the 3rd"
    m = re.search(r"option[^0-9]{0,8}(\d+)", text, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_options:
            return idx
    m = re.search(r"\(\$?([\d\.]+)\s*,\s*\$?([\d\.]+)\)", text)
    if m:
        # Caller can resolve via payoff matching.
        return None
    nums = _all_numbers(text)
    if nums:
        idx = int(nums[0]) - 1
        if 0 <= idx < num_options:
            return idx
    return None


def discrete_to_allocation(idx: int, task: BudgetTask) -> Tuple[float, float]:
    """Convert option index 0..10 to (xA, xB) in points."""
    xA_pts = 10 * idx
    xB_pts = 100 - xA_pts
    return float(xA_pts), float(xB_pts)


def parse_response(text: str,
                   condition: str,
                   task: BudgetTask) -> Optional[Tuple[float, float]]:
    """High-level parser. Returns (xA_points, xB_points) or None on failure."""
    if condition in ("baseline", "price_framing"):
        return parse_continuous(text)
    if condition == "discrete_choice":
        idx = parse_discrete(text)
        if idx is not None:
            return discrete_to_allocation(idx, task)
        # Fallback: try to find a payoff pair and reverse-map
        m = re.search(r"\(\$?([\d\.]+)\s*,\s*\$?([\d\.]+)\)", text)
        if m:
            payA, payB = float(m.group(1)), float(m.group(2))
            for i in range(11):
                xA = 10 * i
                xB = 100 - xA
                if (abs(xA * task.M - payA) < 0.05 and
                        abs(xB * task.N - payB) < 0.05):
                    return float(xA), float(xB)
        return None
    raise ValueError(f"Unknown condition: {condition}")
