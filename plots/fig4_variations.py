"""Figure 4: mean CCEI with 95% CI across robustness variations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VARIATION_LABELS = [
    "baseline", "temp5", "temp1", "priceframing", "discrete",
    "female", "male", "kid", "elder", "low_edu", "high_edu",
    "black", "asian",
]
DISPLAY_LABELS = [
    "Baseline", "Temp=0.5", "Temp=1.0", "Price\nFraming", "Discrete\nChoice",
    "Female", "Male", "Kid", "Elder",
    "Elementary", "College", "African\nAmerican", "Asian",
]
DOMAIN_STYLE = {
    "risk":   ("^", "#1f77b4"),
    "time":   ("o", "brown"),
    "social": ("s", sns.dark_palette("limegreen")[4]),
    "food":   ("D", "purple"),
}


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), 0.0
    mean = float(np.mean(values))
    ci = 1.96 * float(np.std(values, ddof=1) /
                      np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, ci


def make_figure(data_path: Path, out_path: Path,
                model: str = "gpt-3.5-turbo") -> Path:
    df = pd.read_csv(data_path)
    df = df[df["model"] == model]
    if df.empty:
        raise RuntimeError(f"No rows for model={model} in {data_path}")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    n_var = len(VARIATION_LABELS)

    for di, (domain, (marker, color)) in enumerate(DOMAIN_STYLE.items()):
        col = f"ccei_{domain}"
        means, cis = [], []
        for var in VARIATION_LABELS:
            label = f"GPT_{domain}_{var}"
            vals = df.loc[df["label"] == label, col].dropna().values
            mean, ci = _mean_ci(vals)
            means.append(mean)
            cis.append(ci)
        x = np.arange(n_var) * 4 + di
        ax.errorbar(x=x, y=means, yerr=cis, fmt=marker,
                    capsize=5, color=color, markersize=10,
                    label=domain.capitalize(), alpha=1)

    # Tick positions: center of each variation cluster
    tick_x = np.arange(n_var) * 4 + 1.5
    ax.set_xticks(tick_x, DISPLAY_LABELS, rotation=45, fontsize=11)
    ax.set_ylim(0.55, 1.05)
    ax.set_xlim(-0.5, n_var * 4 - 0.5)
    ax.set_ylabel("CCEI", fontsize=14)

    # Vertical separators between clusters
    for i in range(n_var - 1):
        ax.axvline(x=i * 4 + 3.5, linestyle="--", color="gray", alpha=0.5)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=4, fontsize=14)
    ax.set_title(f"Mean CCEI by Variation — {model}", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    args = parser.parse_args()
    print(make_figure(args.data, args.out, args.model))


if __name__ == "__main__":
    main()
