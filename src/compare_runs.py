"""Comparison table of every member's MLflow runs, for the model review (PBI-08.1).

    python -m src.compare_runs                     # runs from the most recent batch
    python -m src.compare_runs --batch base        # runs from one batch
    python -m src.compare_runs --all               # every finished run
    python -m src.compare_runs --csv runs.csv      # also save the table

Prints a Markdown table ranked by test PR-AUC, ready to paste into the GitHub issue that records the
review decision (PBI-08.4). Precision and the false-alarm rate are shown next to recall, because a
model that catches more fraud by flagging many genuine customers isn't better for Trust & Safety.
Test scores use the threshold each run chose on its validation data.
"""
from __future__ import annotations

import argparse
import sys

import mlflow
import pandas as pd

from src.train_common import EXPERIMENT, connect_mlflow

# (MLflow column, heading, number format)
COLUMNS = [
    ("tags.mlflow.runName", "Run", None),
    ("tags.member", "Member", None),
    ("tags.model", "Model", None),
    ("metrics.test_pr_auc", "Test PR-AUC", "{:.3f}"),
    ("metrics.test_precision", "Precision", "{:.3f}"),
    ("metrics.test_recall", "Recall", "{:.3f}"),
    ("metrics.test_f1", "F1", "{:.3f}"),
    ("metrics.test_false_alarm_rate", "False alarms", "{:.1%}"),
    ("metrics.threshold", "Threshold", "{:.3f}"),
    ("metrics.val_pr_auc", "Val PR-AUC", "{:.3f}"),
]


def fetch_runs() -> pd.DataFrame:
    """Every finished run in the team experiment."""
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string="attributes.status = 'FINISHED'")
    if runs.empty:
        raise RuntimeError(f"No finished runs in the MLflow experiment '{EXPERIMENT}'.")
    return runs


def pick_batch(runs: pd.DataFrame, batch: str | None) -> tuple[pd.DataFrame, str]:
    """Runs from one batch: the one asked for, or the batch of the most recent run."""
    if "tags.batch" not in runs or runs["tags.batch"].isna().all():
        raise RuntimeError("Runs have no 'batch' tag; were they logged with src/train_common.py?")
    if batch is None:
        batch = runs.sort_values("start_time")["tags.batch"].dropna().iloc[-1]
    chosen = runs[runs["tags.batch"] == batch]
    if chosen.empty:
        known = ", ".join(sorted(runs["tags.batch"].dropna().unique()))
        raise RuntimeError(f"No runs for batch '{batch}'. Batches in MLflow: {known}")
    return chosen, batch


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run, best test PR-AUC first. If a run name appears twice, the newest run is kept."""
    runs = runs.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
    table = pd.DataFrame({heading: runs[col] if col in runs else pd.NA for col, heading, _ in COLUMNS})
    table = table.sort_values("Test PR-AUC", ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def to_markdown(table: pd.DataFrame) -> str:
    formats = {heading: fmt for _, heading, fmt in COLUMNS}

    def cell(heading, value):
        if pd.isna(value):
            return "-"
        fmt = formats.get(heading)
        return fmt.format(value) if fmt else str(value)

    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(h, row[h]) for h in table.columns) + " |")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare every member's MLflow runs for the model review.")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--batch", help="Only runs from this batch (default: the most recent batch)")
    which.add_argument("--all", action="store_true", help="Every finished run, all batches")
    p.add_argument("--csv", help="Also save the table to this CSV file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        connect_mlflow()
        runs = fetch_runs()
        if args.all:
            title = "all batches"
        else:
            runs, batch = pick_batch(runs, args.batch)
            title = f"batch '{batch}'"
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    table = comparison_table(runs)
    print(f"Model comparison, {title}: {len(table)} runs, ranked by test PR-AUC\n")
    print(to_markdown(table))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nSaved to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
