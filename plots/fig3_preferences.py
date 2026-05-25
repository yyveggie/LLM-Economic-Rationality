"""Figure 3: scatter of estimated (alpha, rho) by domain."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .fig1_ccei import DOMAIN_COLORS, DOMAIN_TITLES


GREEK = {"risk": r"r", "time": r"t", "social": r"s", "food": r"f"}


def _plot_domain(ax, df_models: dict, domain: str):
    color = DOMAIN_COLORS[domain]
    sub_g = GREEK[domain]
    markers = ["s", "o", "^", "D", "v", "P", "X"]
    for i, (model_key, df) in enumerate(df_models.items()):
        sub = df[df["label"] == f"GPT_{domain}_baseline"]
        sub = sub[(sub[f"alpha_{domain}"].notna()) &
                  (sub[f"ccei_{domain}"] > 0.95)]
        if sub.empty:
            continue
        ax.scatter(sub[f"alpha_{domain}"].round(2),
                   sub[f"rho_{domain}"].round(2),
                   marker=markers[i % len(markers)],
                   facecolors=color if i == 0 else "none",
                   edgecolors=color,
                   label=model_key, s=100, alpha=0.75, linewidths=1.5)
    ax.legend(loc=(0.025, 0.025), prop={"size": 12})
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1.0],
                  [r"$\leq$-1", "-0.5", "0.0", "0.5", "1.0"])
    ax.set_xlabel(rf"$\alpha_{sub_g}$", fontsize=22)
    ax.set_ylabel(rf"$\rho_{sub_g}$", fontsize=22)
    ax.text(0.5, -0.25, DOMAIN_TITLES[domain], ha="center", va="center",
            transform=ax.transAxes, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=14)


def make_figure(data_path: Path, out_path: Path,
                models: list[str] | None = None) -> Path:
    df = pd.read_csv(data_path)
    if models:
        df = df[df["model"].isin(models)]
    df_models = {k: g for k, g in df.groupby("model")}

    fig, axs = plt.subplots(2, 2, figsize=(13, 12), dpi=200)
    for ax, domain in zip(axs.flat, ["risk", "time", "social", "food"]):
        _plot_domain(ax, df_models, domain)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
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
