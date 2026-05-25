"""Figure 1: CDFs of CCEI for the four preference domains.

Replaces the original `Fig. 1.ipynb`. Compatible with the schema produced by
`src/analysis.py::build_data_csv`.

Usage:
    python -m plots.fig1_ccei \
        --data results/<exp>/data.csv \
        --models gpt-3.5-turbo gpt-4o \
        --out results/<exp>/Figure1.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DOMAIN_COLORS = {
    "risk":   "#1f77b4",
    "time":   "brown",
    "social": sns.dark_palette("limegreen")[4],
    "food":   "purple",
}
DOMAIN_TITLES = {
    "risk": "Risk Preference",
    "time": "Time Preference",
    "social": "Social Preference",
    "food": "Food Preference",
}


def _plot_domain(ax, df_models: dict, domain: str, sim: pd.Series | None):
    color = DOMAIN_COLORS[domain]
    col = f"ccei_{domain}"
    for model_key, df in df_models.items():
        sub = df[df["label"] == f"GPT_{domain}_baseline"]
        if not sub.empty:
            sns.ecdfplot(data=sub, x=col, lw=2, color=color, ax=ax,
                         label=model_key)
    if sim is not None and len(sim) > 0:
        sns.ecdfplot(data=sim, lw=2, linestyle="dotted", color="grey",
                     ax=ax, label="Simulated")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0, 1.01)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.text(0.5, -0.18, DOMAIN_TITLES[domain], ha="center", va="center",
            transform=ax.transAxes, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12)


def make_figure(data_path: Path, out_path: Path,
                models: list[str] | None = None,
                sim_path: Path | None = None) -> Path:
    df = pd.read_csv(data_path)
    if models:
        df = df[df["model"].isin(models)]
    df_models = {k: g for k, g in df.groupby("model")}

    sim = None
    if sim_path and sim_path.exists():
        sim_df = pd.read_csv(sim_path)
        if "ccei_sim" in sim_df.columns:
            sim = sim_df["ccei_sim"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    for ax, domain in zip(axs.flat, ["risk", "time", "social", "food"]):
        _plot_domain(ax, df_models, domain, sim)

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
    parser.add_argument("--sim", type=Path, default=None)
    args = parser.parse_args()
    print(make_figure(args.data, args.out, args.models, args.sim))


if __name__ == "__main__":
    main()
