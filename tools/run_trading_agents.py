#!/usr/bin/env python
"""
SmartFolio TradingAgents Runner
=================================

This orchestrator executes the Fundamental, News, and Combined agents
to produce explainability summaries for a portfolio allocation CSV.

Usage:
  python tools/run_trading_agents.py \
      --allocation-csv allocation.csv \
      --include-components \
      --print-summaries \
      --llm \
      --llm-model gemini-2.0-flash
"""

from __future__ import annotations
import argparse
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.markdown import Markdown

# ---------------------------------------------------------------------
# 🔧 Fix PYTHONPATH so the script finds trading_agent/tradingagents/
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # /SmartFolio
AGENT_ROOT = ROOT / "trading_agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

# ---------------------------------------------------------------------
# ✅ Import TradingAgents
# ---------------------------------------------------------------------
from tradingagents.combined_weight_agent import WeightSynthesisAgent
from tradingagents.fundamental_agent import FundamentalWeightAgent
from tradingagents.news_agent import NewsWeightReviewAgent

console = Console()


# ---------------------------------------------------------------------
# 📦 Monthly Log Helpers
# ---------------------------------------------------------------------
def build_monthly_snapshot(csv_path: Path, sample_day: int = 10, run_id: str | None = None) -> pd.DataFrame:
    """Create a per-month allocation slice from a final_test_weights CSV."""

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "date" not in df.columns:
        raise ValueError("monthly log CSV must contain a 'date' column")
    if "weight" not in df.columns:
        raise ValueError("monthly log CSV must contain a 'weight' column")

    if run_id and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
        if df.empty:
            raise ValueError(f"No rows found for run_id={run_id} in {csv_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("One or more rows in monthly log CSV have invalid dates")

    df["month"] = df["date"].dt.to_period("M")
    snapshots: list[pd.DataFrame] = []
    for _, group in df.groupby("month"):
        group_sorted = group.sort_values("date")
        preferred = group_sorted[group_sorted["date"].dt.day == sample_day]
        if preferred.empty:
            preferred = group_sorted[group_sorted["date"].dt.day > sample_day]
        if preferred.empty:
            preferred = group_sorted
        chosen_date = preferred.iloc[0]["date"]
        slice_df = group[group["date"] == chosen_date].copy()
        slice_df["as_of"] = chosen_date.strftime("%Y-%m-%d")
        snapshots.append(slice_df[[col for col in slice_df.columns if col in {"ticker", "weight", "as_of"}]])

    if not snapshots:
        raise ValueError("No monthly slices could be derived from the provided log CSV")

    combined = pd.concat(snapshots, ignore_index=True)
    return combined


def select_top_holdings(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Return the top-K holdings by weight (per as_of date when present)."""

    if top_k is None or top_k <= 0 or "weight" not in df.columns:
        return df

    def _pick(group: pd.DataFrame) -> pd.DataFrame:
        return group.sort_values("weight", ascending=False).head(top_k)

    if "as_of" in df.columns:
        filtered = (
            df.groupby("as_of", group_keys=False, sort=False)
            .apply(_pick)
            .reset_index(drop=True)
        )
    else:
        filtered = _pick(df)
    return filtered


# ---------------------------------------------------------------------
# 🔹 CLI Argument Parser
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run SmartFolio TradingAgents suite.")
    p.add_argument(
        "--allocation-csv",
        default="allocation.csv",
        help="CSV file with at least columns: ticker,weight (default: allocation.csv)",
    )
    p.add_argument(
        "--monthly-log-csv",
        default=None,
        help="Optional final_test_weights CSV from logs/monthly. When provided, overrides --allocation-csv.",
    )
    p.add_argument(
        "--monthly-sample-day",
        type=int,
        default=10,
        help="Preferred calendar day to sample per month when using --monthly-log-csv (default: 10).",
    )
    p.add_argument(
        "--monthly-run-id",
        default=None,
        help="Optional run_id filter for --monthly-log-csv (defaults to all run_ids present).",
    )
    p.add_argument(
        "--trading-agent-root",
        default=str(AGENT_ROOT),
        help="Path to trading_agent package (default: trading_agent)",
    )
    p.add_argument(
        "--include-components",
        action="store_true",
        help="Include detailed fundamentals and news components in markdown output.",
    )
    p.add_argument(
        "--include-metrics",
        action="store_true",
        help="Include the fundamental metrics table in the report.",
    )
    p.add_argument(
        "--include-articles",
        action="store_true",
        help="Include news headline tables in the report.",
    )
    p.add_argument(
        "--print-summaries",
        action="store_true",
        help="Print markdown reports directly to console.",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM synthesis via Gemini 2.0 Flash for the unified summary.",
    )
    p.add_argument(
        "--llm-model",
        default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel threads to run agent computations.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-weight tickers to explain (per as_of snapshot when available).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# 🧩 Worker Function for Each Ticker
# ---------------------------------------------------------------------
def run_agent_for_row(row, args):
    ticker = str(row["ticker"]).strip().upper()
    weight = float(row["weight"])
    as_of = None
    if "as_of" in row and isinstance(row["as_of"], str):
        as_of = row["as_of"]
    try:
        agent = WeightSynthesisAgent()
        report = agent.generate_report(
            ticker,
            weight,
            as_of=as_of,
            lookback_days=30,
            max_articles=8,
            use_llm=args.llm,
            llm_model=args.llm_model,
        )

        markdown_text = report.to_markdown(
            include_components=args.include_components,
            include_metrics=args.include_metrics,
            include_articles=args.include_articles,
        )

        # Save to explainability_results/
        out_dir = Path("explainability_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_summary.md"
        out_path.write_text(markdown_text, encoding="utf-8")

        return ticker, True, report.generated_via_llm, out_path, markdown_text
    except Exception as e:
        return ticker, False, False, None, f"[ERROR] {ticker}: {e}"


# ---------------------------------------------------------------------
# 🚀 Main Orchestration Function
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    requested_root = Path(args.trading_agent_root).expanduser().resolve()
    if requested_root != AGENT_ROOT.resolve() and str(requested_root) not in sys.path:
        sys.path.insert(0, str(requested_root))

    def _load_allocations() -> pd.DataFrame:
        if args.monthly_log_csv:
            log_path = Path(args.monthly_log_csv)
            if not log_path.exists():
                console.print(f"[red]❌ Missing monthly log CSV: {log_path}[/red]")
                sys.exit(1)
            return build_monthly_snapshot(
                log_path,
                sample_day=args.monthly_sample_day,
                run_id=args.monthly_run_id,
            )

        csv_path = Path(args.allocation_csv)
        if not csv_path.exists():
            console.print(f"[red]❌ Missing allocation CSV: {csv_path}[/red]")
            sys.exit(1)
        df_csv = pd.read_csv(csv_path)
        df_csv["as_of"] = df_csv.get("as_of")
        return df_csv

    df = _load_allocations()
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "ticker" not in df.columns or "weight" not in df.columns:
        console.print("[red]❌ CSV must include columns 'ticker' and 'weight'[/red]")
        sys.exit(1)

    df = select_top_holdings(df, args.top_k)
    if df.empty:
        console.print("[red]❌ No rows left after filtering top holdings.[/red]")
        sys.exit(1)

    console.print(
        f"[bold cyan]🚀 Running TradingAgents for {len(df)} tickers...[/bold cyan]\n"
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_agent_for_row, row, args): row for _, row in df.iterrows()}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Summary
    success_count = sum(1 for _, ok, *_ in results if ok)
    fail_count = len(results) - success_count
    console.print(f"\n✅ [green]{success_count} succeeded[/green], ❌ [red]{fail_count} failed[/red].\n")

    # Console printing
    if args.print_summaries:
        for ticker, ok, via_llm, path, output in results:
            console.print(f"\n[bold yellow]{ticker}[/bold yellow]")
            if not ok:
                console.print(f"[red]{output}[/red]\n")
                continue
            console.print(Markdown(output))
            console.print(f"[dim]Saved: {path}[/dim]")
            if via_llm:
                console.print("[dim green]Generated via LLM (Gemini 2.0 Flash)[/dim]\n")

    # Write summary index CSV
    summary_data = [
        {
            "ticker": t,
            "success": ok,
            "llm_used": via_llm,
            "output_path": str(p) if p else "",
        }
        for t, ok, via_llm, p, _ in results
    ]
    summary_df = pd.DataFrame(summary_data)
    out_dir = Path("explainability_results")
    out_dir.mkdir(exist_ok=True)
    summary_csv = out_dir / "summary_index.csv"
    summary_df.to_csv(summary_csv, index=False)
    console.print(f"\n📊 [bold green]Summary index written to {summary_csv}[/bold green]\n")


# ---------------------------------------------------------------------
# 🔘 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
