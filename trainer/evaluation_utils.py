"""Utility helpers for evaluation logging and promotion decisions."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PromotionDecision:
    """Container describing the outcome of a promotion gate."""

    promoted: bool
    reasons: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {"promoted": self.promoted, "reasons": self.reasons}


def _determine_log_timestamp(args: object) -> pd.Timestamp:
    """Derive a timestamp to use for log folder selection."""
    candidate_attrs = [
        "test_end_date",
        "val_end_date",
        "train_end_date",
    ]
    for attr in candidate_attrs:
        value = getattr(args, attr, None)
        if value:
            try:
                return pd.Timestamp(value)
            except (TypeError, ValueError):
                continue
    return pd.Timestamp(datetime.utcnow())


def _monthly_log_dir(base_dir: Path, timestamp: pd.Timestamp) -> Path:
    month_str = timestamp.strftime("%Y-%m")
    target = base_dir / month_str
    target.mkdir(parents=True, exist_ok=True)
    return target


def _split_date_range(args: object, split: str) -> Tuple[Optional[str], Optional[str]]:
    split_lower = split.lower()
    if "val" in split_lower:
        start = getattr(args, "val_start_date", None)
        end = getattr(args, "val_end_date", None)
    elif "train" in split_lower:
        start = getattr(args, "train_start_date", None)
        end = getattr(args, "train_end_date", None)
    else:
        start = getattr(args, "test_start_date", None)
        end = getattr(args, "test_end_date", None)
    return start, end


def create_metric_record(
    args: object,
    split: str,
    metrics: Dict[str, Optional[float]],
    batch_index: int,
    timestamp: Optional[datetime] = None,
) -> Dict[str, object]:
    """Create a normalized dictionary with metadata for persistence."""
    ts = timestamp or datetime.utcnow()
    run_id = f"{split}_{ts.strftime('%Y%m%d_%H%M%S')}_{batch_index}"
    eval_start, eval_end = _split_date_range(args, split)

    record: Dict[str, object] = {
        "run_id": run_id,
        "created_at": ts.isoformat(),
        "split": split,
        "batch_index": batch_index,
        "market": getattr(args, "market", None),
        "model_name": getattr(args, "model_name", None),
        "policy": getattr(args, "policy", None),
        "evaluation_start": eval_start,
        "evaluation_end": eval_end,
    }
    for key, value in metrics.items():
        record[key] = float(value) if value is not None else None
    return record


def aggregate_metric_records(records: Sequence[Dict[str, object]]) -> Dict[str, Optional[float]]:
    """Aggregate evaluation records into a summary (mean for numeric keys)."""
    if not records:
        return {}
    numeric_keys = ["arr", "avol", "sharpe", "mdd", "cr", "ir"]
    summary: Dict[str, Optional[float]] = {"count": len(records)}
    for key in numeric_keys:
        values: List[float] = []
        for record in records:
            value = record.get(key)
            if value is not None:
                values.append(float(value))
        summary[key] = float(np.mean(values)) if values else None
    return summary


def persist_metrics(
    records: Sequence[Dict[str, object]],
    env_snapshots: Sequence[Tuple[object, str]],
    args: object,
    split: str,
    base_dir: str = "logs/monthly",
) -> Dict[str, object]:
    """Persist metrics to JSON/CSV and export curve diagnostics."""
    if not records:
        return {"records": [], "log_dir": None}

    timestamp = _determine_log_timestamp(args)
    base_path = Path(base_dir)
    log_dir = _monthly_log_dir(base_path, timestamp)

    timestamp_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"{split}_{timestamp_label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, indent=2)

    csv_path = log_dir / "metrics.csv"
    new_frame = pd.DataFrame(list(records))
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame
    combined.to_csv(csv_path, index=False)

    # Export wealth and return curves for each snapshot
    for env_instance, run_id in env_snapshots:
        try:
            df_net = env_instance.get_df_net_value()
            df_ret = env_instance.get_df_daily_return()
        except Exception:
            continue
        net_path = log_dir / f"{run_id}_net_value.csv"
        ret_path = log_dir / f"{run_id}_daily_return.csv"
        df_net.to_csv(net_path, index=False)
        df_ret.to_csv(ret_path, index=False)

    return {
        "log_dir": log_dir,
        "json_path": json_path,
        "csv_path": csv_path,
        "record_ids": [record["run_id"] for record in records],
    }


def should_promote_model(
    summary_metrics: Dict[str, Optional[float]],
    min_sharpe: Optional[float] = None,
    max_drawdown: Optional[float] = None,
) -> PromotionDecision:
    """Evaluate whether a checkpoint should replace the baseline."""
    reasons: List[str] = []
    sharpe = summary_metrics.get("sharpe") if summary_metrics else None
    mdd = summary_metrics.get("mdd") if summary_metrics else None

    if min_sharpe is not None:
        if sharpe is None or sharpe < min_sharpe:
            reasons.append(
                f"Sharpe {sharpe if sharpe is not None else 'N/A'} below minimum {min_sharpe}"
            )
    if max_drawdown is not None:
        # max_drawdown is provided as a positive fraction (e.g., 0.2 for 20%)
        allowed = abs(max_drawdown)
        if mdd is None:
            reasons.append("Drawdown unavailable")
        elif abs(mdd) > allowed:
            reasons.append(f"Drawdown {abs(mdd):.4f} exceeds limit {allowed}")

    return PromotionDecision(promoted=len(reasons) == 0, reasons=reasons)


def log_promotion_event(
    log_dir: Optional[Path],
    decision: PromotionDecision,
    summary_metrics: Dict[str, Optional[float]],
    candidate_path: Optional[str],
    baseline_path: Optional[str],
) -> None:
    """Persist promotion decisions alongside metrics."""
    if log_dir is None:
        return
    record = {
        "created_at": datetime.utcnow().isoformat(),
        "promoted": decision.promoted,
        "reasons": decision.reasons,
        "candidate_path": candidate_path,
        "baseline_path": baseline_path,
    }
    record.update(summary_metrics or {})

    json_path = log_dir / "promotion_events.json"
    try:
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
    except json.JSONDecodeError:
        existing = []
    existing.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    csv_path = log_dir / "promotion_events.csv"
    df_new = pd.DataFrame([record])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)


def apply_promotion_gate(
    args: object,
    candidate_path: Optional[str],
    summary_metrics: Dict[str, Optional[float]],
    log_info: Optional[Dict[str, object]] = None,
) -> PromotionDecision:
    """Run gating logic and copy the checkpoint if approved."""
    min_sharpe = getattr(args, "promotion_min_sharpe", None)
    max_drawdown = getattr(args, "promotion_max_drawdown", None)
    baseline_path = getattr(args, "baseline_checkpoint", None)

    decision = should_promote_model(summary_metrics, min_sharpe, max_drawdown)
    log_dir = None
    if log_info:
        log_dir = log_info.get("log_dir")

    if not candidate_path:
        log_promotion_event(log_dir, decision, summary_metrics, candidate_path, baseline_path)
        return decision

    if not baseline_path:
        print("Baseline checkpoint path not provided; skipping promotion gate.")
        log_promotion_event(log_dir, decision, summary_metrics, candidate_path, baseline_path)
        return decision

    baseline_path = str(baseline_path)
    if decision.promoted:
        try:
            target_path = Path(baseline_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate_path, baseline_path)
            print(f"Promoted new baseline checkpoint: {baseline_path}")
        except Exception as exc:
            reason = f"Failed to promote checkpoint: {exc}"
            print(reason)
            decision = PromotionDecision(False, decision.reasons + [reason])
    else:
        print(
            "Skipping promotion of fine-tuned checkpoint due to gating criteria: "
            + "; ".join(decision.reasons)
        )

    log_promotion_event(log_dir, decision, summary_metrics, candidate_path, baseline_path)
    return decision
