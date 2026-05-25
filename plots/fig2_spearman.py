"""Figure 2: CDFs of Spearman correlation between ln(xA/xB) and ln(pA/pB)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .fig1_ccei import DOMAIN_COLORS, DOMAIN_TITLES


def _plot_domain(ax, df_models: dict, domain: str):
    color = DOMAIN_COLORS[domain]
    col = f"spearman_{domain}"
    for model_key, df in df_models.items():
        sub = df[df["label"] == f"GPT_{domain}_baseline"]
        if not sub.empty:
            sns.ecdfplot(data=sub, x=col, lw=2, color=color, ax=ax,
                         label=model_key)
    ax.set_xlim(-1.05, 0.3)
    ax.set_ylim(0, 1.01)
    ax.set_xticks(np.arange(-1, 0.5, 0.25))
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.text(0.5, -0.18, DOMAIN_TITLES[domain], ha="center", va="center",
            transform=ax.transAxes, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12, loc="lower right")


def make_figure(data_path: Path, out_path: Path,
                models: list[str] | None = None) -> Path:
    df = pd.read_csv(data_path)
    if models:
        df = df[df["model"].isin(models)]
    df_models = {k: g for k, g in df.groupby("model")}

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    for ax, domain in zip(axs.flat, ["risk", "time", "social", "food"]):
        _plot_domain(ax, df_models, domain)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    print(make_figure(args.data, args.out, args.models))


if __name__ == "__main__":
    main()
