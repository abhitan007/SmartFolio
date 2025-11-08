"""Aggregate monthly evaluation metrics for monitoring."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


NUMERIC_COLUMNS: List[str] = ["arr", "avol", "sharpe", "mdd", "cr", "ir"]


def load_monthly_frames(base_dir: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for month_dir in sorted(base_dir.glob("*/")):
        metrics_path = month_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        df["month"] = month_dir.name.rstrip("/")
        frames.append(df)
    return frames


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["month", "market", "policy", "split"]
    available_numeric = [col for col in NUMERIC_COLUMNS if col in df.columns]
    agg_dict = {col: "mean" for col in available_numeric}
    agg_dict.update({"run_id": "count"})
    summary = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    summary = summary.rename(columns={"run_id": "evaluations"})
    return summary


def main() -> None:
    base_dir = Path("logs/monthly")
    if not base_dir.exists():
        print("No monthly logs found.")
        return

    frames = load_monthly_frames(base_dir)
    if not frames:
        print("No metrics available to summarise.")
        return

    combined = pd.concat(frames, ignore_index=True)
    summary = compute_summary(combined)

    output_path = base_dir / "monthly_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Wrote monthly summary to {output_path}")
    print(summary)


if __name__ == "__main__":
    main()
