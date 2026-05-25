"""Aggregate raw LLM decisions into the data.csv-style metrics table."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .estimation import estimate
from .rationality import Decision, all_indices


log = logging.getLogger(__name__)
DOMAINS = ["risk", "time", "social", "food"]


def _to_decisions(df: pd.DataFrame) -> List[Decision]:
    return [Decision(pA=row.pA, pB=row.pB, xA=row.xA, xB=row.xB)
            for row in df.itertuples(index=False)]


def aggregate_subject_level(raw_csv: Path) -> pd.DataFrame:
    """One row per (model, label, subject), with metrics for the matching domain.

    For compatibility with the original `data.csv`, columns from other domains
    are filled with NaN; the `label` already encodes which domain is active.
    """
    raw = pd.read_csv(raw_csv)
    raw = raw.dropna(subset=["xA", "xB"])
    records: List[Dict] = []

    grouped = raw.groupby(["model", "label", "subject"], sort=False)
    for (model, label, subject), grp in grouped:
        if len(grp) < 2:
            continue
        # All rows in a group share the same domain, by construction
        domain = grp["domain"].iloc[0]
        decisions = _to_decisions(grp[["pA", "pB", "xA", "xB"]])
        idx = all_indices(decisions)
        est = estimate(decisions, domain)

        rec: Dict = {"model": model, "label": label, "subject": subject}
        rec[f"ccei_{domain}"] = idx["ccei"]
        rec[f"hmi_{domain}"] = idx["hmi"]
        rec[f"mpi_{domain}"] = idx["mpi"]
        rec[f"mci_{domain}"] = idx["mci"]
        rec[f"spearman_{domain}"] = idx["spearman"]
        rec[f"alpha_{domain}"] = est["alpha"]
        rec[f"rho_{domain}"] = est["rho"]
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Fill missing per-domain columns so the schema matches data.csv
    for d in DOMAINS:
        for col in (f"ccei_{d}", f"hmi_{d}", f"mpi_{d}", f"mci_{d}",
                    f"spearman_{d}", f"alpha_{d}", f"rho_{d}"):
            if col not in df.columns:
                df[col] = np.nan

    log.info("Subject-level table: %d rows", len(df))
    return df


def aggregate_population_level(subject_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, label) means / SEs for the four CES parameters."""
    rows = []
    for (model, label), grp in subject_df.groupby(["model", "label"]):
        domain = label.split("_")[1]  # GPT_<domain>_<variant>
        alpha = grp[f"alpha_{domain}"].dropna()
        rho = grp[f"rho_{domain}"].dropna()
        rows.append({
            "model": model,
            "label": label,
            f"alpha_{domain}_agg": alpha.mean() if len(alpha) else np.nan,
            f"rho_{domain}_agg": rho.mean() if len(rho) else np.nan,
            f"alpha_se_{domain}_agg": (alpha.std(ddof=1) / np.sqrt(len(alpha))
                                       if len(alpha) > 1 else np.nan),
            f"rho_se_{domain}_agg": (rho.std(ddof=1) / np.sqrt(len(rho))
                                     if len(rho) > 1 else np.nan),
        })
    return pd.DataFrame(rows)


def build_data_csv(raw_csvs: List[Path], output_csv: Path) -> Path:
    """Concatenate per-model raw CSVs, compute metrics, write data.csv."""
    pieces = [aggregate_subject_level(p) for p in raw_csvs if p.exists()]
    if not pieces:
        raise RuntimeError("No raw decision files found.")
    subj = pd.concat(pieces, ignore_index=True)
    pop = aggregate_population_level(subj)
    merged = subj.merge(pop, on=["model", "label"], how="left")
    merged.insert(0, "id", range(len(merged)))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    log.info("Wrote aggregated metrics to %s (%d rows)", output_csv, len(merged))
    return output_csv
