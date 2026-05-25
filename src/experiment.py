"""Drive the LLM through the budgetary tasks and persist raw decisions."""
from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .llm_client import LLMClient, ModelConfig
from .parser import parse_response
from .prompts import build_prompt
from .tasks import BudgetTask, generate_tasks


log = logging.getLogger(__name__)


# Mirrors the original data.csv: temp5 == 0.5, temp1 == 1.0.
# Any other non-zero temperature falls back to "tempXX" (XX = round(temp*100)).
_TEMP_LABEL = {0.5: "temp5", 1.0: "temp1"}


def _label(domain: str, condition: str, demographic: str, temperature: float) -> str:
    """Stable label used for grouping results, mirrors the original CSV."""
    parts = [domain]
    if condition == "baseline":
        if demographic == "none":
            if temperature == 0.0:
                parts.append("baseline")
            else:
                parts.append(_TEMP_LABEL.get(float(temperature),
                                             f"temp{int(round(temperature * 100))}"))
        else:
            parts.append(demographic)
    elif condition == "price_framing":
        parts.append("priceframing")
    elif condition == "discrete_choice":
        parts.append("discrete")
    return "GPT_" + "_".join(parts)


def run_one_subject(client: LLMClient,
                    domain: str,
                    condition: str,
                    demographic: str,
                    temperature: float,
                    tasks: List[BudgetTask],
                    save_raw: bool) -> List[Dict]:
    rows: List[Dict] = []
    for task in tasks:
        prompt = build_prompt(domain, condition, task, demographic)
        try:
            text = client.complete(
                system=prompt.system,
                messages=[
                    {"role": "assistant", "content": prompt.assistant},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=temperature,
            )
        except Exception as exc:
            log.warning("LLM call failed (%s round %d): %s",
                        domain, task.round_idx, exc)
            text = ""

        parsed = parse_response(text, condition, task)
        xA, xB = (parsed if parsed is not None else (float("nan"), float("nan")))
        row = {
            "domain": domain,
            "condition": condition,
            "demographic": demographic,
            "temperature": temperature,
            "round": task.round_idx,
            "M": task.M,
            "N": task.N,
            "pA": task.pA,
            "pB": task.pB,
            "xA": xA,
            "xB": xB,
        }
        if save_raw:
            row["raw_response"] = text
        rows.append(row)
    return rows


def run_experiment_for_model(model_cfg: ModelConfig,
                             experiment: dict,
                             global_cfg: dict,
                             output_root: Path) -> Path:
    """Run all (domain, condition, demographic, temperature) cells for one model."""
    overrides = experiment.get("overrides", {}) or {}
    num_subjects = int(overrides.get("num_subjects",
                                     global_cfg.get("num_subjects", 100)))
    rounds = int(global_cfg.get("rounds_per_subject", 25))
    seed = int(global_cfg.get("random_seed", 20231212))
    save_raw = bool(global_cfg.get("save_raw_responses", True))

    domains = experiment["domains"]
    conditions = experiment["conditions"]
    variations = experiment.get("variations") or {}
    temps = variations.get("temperatures") or [model_cfg.default_temperature]
    demos = variations.get("demographics") or ["none"]

    out_dir = output_root / model_cfg.key
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_decisions.csv"

    # Generate the 25 tasks once per (domain, demographic, temperature, condition)
    # Same seed -> same task sequences across models.
    subject_pools = generate_tasks(num_subjects, rounds, seed=seed)

    client = LLMClient(model_cfg)

    # Fresh run overwrites any previous file. Use --skip-llm to reuse it.
    write_header = True
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer: Optional[csv.DictWriter] = None

        cells = []
        for domain in domains:
            for condition in conditions:
                # price_framing/discrete_choice 不再做 demographic / temp 变体
                if condition != "baseline":
                    cells.append((domain, condition, "none",
                                  model_cfg.default_temperature))
                else:
                    for temp in temps:
                        for demo in demos:
                            cells.append((domain, condition, demo, float(temp)))

        for domain, condition, demo, temp in cells:
            label = _label(domain, condition, demo, temp)
            log.info("[%s] running %s (%d subjects)",
                     model_cfg.key, label, num_subjects)
            with ThreadPoolExecutor(max_workers=model_cfg.concurrency) as pool:
                futures = []
                for subj_idx, tasks in enumerate(subject_pools):
                    futures.append(pool.submit(
                        run_one_subject, client, domain, condition, demo,
                        temp, tasks, save_raw))

                pbar = tqdm(as_completed(futures), total=len(futures),
                            desc=f"{model_cfg.key}/{label}", leave=False)
                for subj_idx, fut in enumerate(pbar):
                    rows = fut.result()
                    for row in rows:
                        row["model"] = model_cfg.key
                        row["label"] = label
                        row["subject"] = subj_idx
                    if writer is None:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                    writer.writerows(rows)
                    f.flush()

    log.info("[%s] raw decisions written to %s", model_cfg.key, raw_path)
    return raw_path
