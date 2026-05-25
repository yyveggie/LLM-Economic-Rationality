"""One-stop entry point for the replication study.

Examples
--------
# 1) Smoke test (uses the model defined in configs/models.yaml)
python run.py --experiment smoke_test

# 2) Reproduce Figure 4 robustness panel
python run.py --experiment framing_sensitivity --plots

# 3) Skip LLM calls, only re-run analysis + plots
python run.py --experiment baseline_quick --skip-llm --plots

To compare multiple models, edit configs/models.yaml, run the experiment,
then change models.yaml again and re-run. Each model's results land in
results/<experiment>/<model_key>/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import yaml

# allow `python run.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis import build_data_csv  # noqa: E402
from src.experiment import run_experiment_for_model  # noqa: E402
from src.llm_client import ModelConfig  # noqa: E402


REPO = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_plots(data_csv: Path, out_dir: Path,
              model_key: str, sim_path: Path | None) -> None:
    from plots.fig1_ccei import make_figure as fig1
    from plots.fig2_spearman import make_figure as fig2
    from plots.fig3_preferences import make_figure as fig3
    from plots.fig4_variations import make_figure as fig4

    out_dir.mkdir(parents=True, exist_ok=True)
    models = [model_key]
    fig1(data_csv, out_dir / "Figure1_CCEI.pdf", models, sim_path)
    fig2(data_csv, out_dir / "Figure2_Spearman.pdf", models)
    fig3(data_csv, out_dir / "Figure3_Preferences.pdf", models)
    fig4(data_csv, out_dir / f"Figure4_Variations_{model_key}.pdf", model_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM economic-rationality replication.")
    parser.add_argument("--experiment", help="Override active experiment name.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM calls; reuse existing raw_decisions.csv.")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip metrics computation.")
    parser.add_argument("--plots", action="store_true",
                        help="Render figures after analysis.")
    parser.add_argument("--models-config", default=REPO / "configs" / "models.yaml",
                        type=Path)
    parser.add_argument("--experiments-config",
                        default=REPO / "configs" / "experiments.yaml", type=Path)
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiments_config)
    mod_raw = load_yaml(args.models_config)
    model_cfg = ModelConfig.from_yaml(mod_raw)

    exp_name = args.experiment or exp_cfg.get("active")
    if not exp_name:
        raise RuntimeError("No experiment selected.")
    if exp_name not in exp_cfg["experiments"]:
        raise KeyError(f"Experiment '{exp_name}' not in experiments.yaml")
    experiment = exp_cfg["experiments"][exp_name]
    global_cfg = exp_cfg.get("global", {}) or {}

    setup_logging(global_cfg.get("log_level", "INFO"))
    log = logging.getLogger("run")
    log.info("Experiment: %s", exp_name)
    log.info("  description: %s", experiment.get("description", ""))
    log.info("Model: %s (provider=%s)", model_cfg.model, model_cfg.provider)

    output_root = REPO / global_cfg.get("output_dir", "results") / exp_name
    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv = output_root / model_cfg.key / "raw_decisions.csv"

    if not args.skip_llm:
        run_experiment_for_model(model_cfg, experiment, global_cfg, output_root)
    else:
        log.info("[skip-llm] reusing %s", raw_csv)

    data_csv = output_root / "data.csv"
    if not args.skip_analysis:
        build_data_csv([raw_csv], data_csv)

    if args.plots:
        if not data_csv.exists():
            raise RuntimeError("data.csv missing; run analysis first.")
        sim_path = REPO / "power" / "garp_sim.csv"  # optional
        run_plots(data_csv, output_root / "figures", model_cfg.key,
                  sim_path if sim_path.exists() else None)
        log.info("Figures written to %s", output_root / "figures")

    log.info("Done.")


if __name__ == "__main__":
    main()
