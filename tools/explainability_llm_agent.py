#!/usr/bin/env python
"""
Explainability narrative generator for SmartFolio.

Optimized for Gemini 2.0 Flash:
- Focuses on per-stock logic (r², avg_weight, feature_importances, rules)
- Structured, concise, analytical language
- No fluff or decorative text

Usage:
  export GOOGLE_API_KEY="YOUR_KEY"
  python tools/explainability_llm_agent.py --llm --print
"""

from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np
import joblib
import google.generativeai as genai


# ---------------------------
# CORE CONFIG
# ---------------------------
SYSTEM_PROMPT = ("""
You are a Senior Portfolio Analyst for SmartFolio. Your goal is to explain the "Why" behind the AI's trading decisions to a client.

### Input Data
You will receive a set of decision rules derived from a Surrogate Decision Tree. 
Because the system has already mapped technical indices to names, your input will look like this:
- **Target**: LUPIN.NS
- **Rule**: If AARTIIND.NS (Volume) > 4.88 AND BATAINDIA.NS (Close) <= 4.18 -> Weight: 14%

### Your Task
Translate these structured rules into a qualitative financial narrative.

### Guidelines
1.  **Interpret the Signals**:
    - "Volume" > High Threshold implies high market activity or potential breakout.
    - "Prev_Close" or "Momentum" > High Threshold implies trend following.
    - "Close" < Low Threshold might imply "buy the dip" or mean reversion.
    
2.  **Identify Relationships**:
    - If the rule for buying Stock A depends on Stock B (e.g., buying LUPIN based on AARTIIND's volume), describe this as a **Cross-Market Signal**.
    - Explain *why* the model might be doing this (e.g., "Sector rotation," "Supply chain correlation," or "Risk-off sentiment").

3.  **Tone**: Professional, insightful, and data-driven, but accessible. Avoid raw tree output syntax (like `|---`).

### Output Structure
Produce a Markdown report with these sections:

#### 1. Executive Summary
A single sentence summary of the position's primary driver.
*(Example: "The heavy allocation to Lupin is driven by a volume breakout in the Chemicals sector, specifically tracking Aarti Industries.")*

#### 2. Key Drivers
- **Primary Signal**: The most critical condition (top of the tree).
- **Context**: Secondary conditions that confirm the trade.

#### 3. Market Narrative
Weave the rules into a short story. 
*(Example: "The model adopts a 'Momentum' stance on Lupin. It waits for high activity in peer stocks (Aarti Ind) but requires Bata India to remain stable/low, effectively hedging against broader consumer volatility.")*
"""
)


@dataclass
class SnapshotContext:
    metadata: Dict[str, str]
    filtered_data: Dict[str, object]


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SmartFolio explainability narrative (Gemini 2.0 Flash).")
    p.add_argument("--snapshot", default="explainability_results/explain_tree_custom.joblib")
    p.add_argument("--llm", action="store_true", help="Enable Gemini generation")
    p.add_argument("--llm-model", default="gemini-2.0-flash", help="LLM model name")
    p.add_argument("--output", default="explainability_results/explainability_narrative.md")
    p.add_argument("--print", action="store_true")
    return p.parse_args()


# ---------------------------
# SAFE CONVERSIONS
# ---------------------------
def safe_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(v) for v in obj]
    else:
        return safe_convert(obj)


# ---------------------------
# LOAD SNAPSHOT
# ---------------------------
def load_snapshot(path: Path) -> SnapshotContext:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    data = joblib.load(path)
    if not isinstance(data, dict):
        raise TypeError("Joblib file must contain a dict.")

    keep = ["per_stock", "avg_weights", "top_indices", "global_r2", "X_shape", "Y_shape"]
    filtered = {k: data.get(k) for k in keep if k in data}

    meta = {"model_path": str(path), "included_keys": list(filtered.keys())}
    return SnapshotContext(metadata=meta, filtered_data=filtered)


# ---------------------------
# PROMPT ENGINEERING
# ---------------------------
def assemble_prompt(ctx: SnapshotContext) -> str:
    """
    Builds a structured Gemini-optimized prompt emphasizing reasoning over format noise.
    """

    example_block = (
        "Example format:\n\n"
        "ICICIBANK.NS (Stock 3)\n"
        "    Model Fit (R²): 0.597 → decent fit\n"
        "    Average Allocation: 0.0119\n"
        "    Key Drivers: Stock[76]::Feature_5 (21.9%), Stock[55]::Feature_2 (18.1%), Stock[81]::Feature_3 (17.3%)\n"
        "    High Allocation Logic: High weight (0.09) when Stock[55]::Feature_2 > 6.12\n"
        "    Secondary Logic: Moderate allocation (0.08) under other high feature paths\n"
        "    Low Allocation Logic: Lowest (0.01) when these features are low\n\n"
    )

    instructions = (
        "You are given structured JSON data summarizing decision-tree surrogates that approximate "
        "an RL allocation model. For each stock (in `per_stock`), generate a clear, technical summary "
        "covering these six points in the same order:\n\n"
        "1. Model Fit (R²): State the numeric R² and qualitatively rate the fit as good (≥0.65), "
        "decent (0.45–0.65), or poor (<0.45).\n"
        "2. Average Allocation: Report the agent’s average portfolio weight.\n"
        "3. Key Drivers: List top 2–4 features by importance percentage from `feature_importances`.\n"
        "4. High Allocation Logic: Interpret the conditions in `rules` that yield the highest allocation value.\n"
        "5. Secondary/Moderate Logic: Summarize alternative paths that produce mid-range weights.\n"
        "6. Low Allocation Logic: Describe conditions that lead to the lowest or zero allocation.\n\n"
        "Rules:\n"
        "- No emojis or decorative language.\n"
        "- Output must be structured exactly as the example block format.\n"
        "- Be concise but specific in interpreting thresholds.\n"
        "- Avoid restating JSON keys; translate them into human analysis.\n"
    )

    payload = {
        "system_instruction": SYSTEM_PROMPT,
        "metadata": ctx.metadata,
        "instructions": instructions,
        "example_output": example_block,
        "explainability_data": ctx.filtered_data,
    }
    print(payload)

    return json.dumps(convert_keys_to_str(payload), indent=2, default=safe_convert)


# ---------------------------
# GEMINI CALL (with retry)
# ---------------------------
def llm_narrative(prompt: str, model="gemini-2.0-flash", retries=3, delay=4) -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY.")
    genai.configure(api_key=key)
    llm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Gemini 2.0 Flash call attempt {attempt}/{retries}")
            resp = llm.generate_content(prompt, generation_config={"temperature": 0.3, "top_p": 0.9})
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            raise RuntimeError("Empty Gemini response")
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < retries:
                print(f"[WARN] Rate limited (429). Retrying in {delay}s…")
                time.sleep(delay)
                continue
            print(f"[ERROR] Gemini call failed: {e}")
            return f"**LLM generation failed:** {e}"
    return "**LLM generation unavailable.**"


# ---------------------------
# FALLBACK
# ---------------------------
def fallback_narrative(ctx: SnapshotContext) -> str:
    d = ctx.filtered_data
    return (
        f"Fallback summary — global R² {d.get('global_r2','n/a')}, "
        f"{len(d.get('per_stock',{}))} stocks analyzed. "
        "Detailed logic requires LLM generation."
    )


# ---------------------------
# MAIN
# ---------------------------
def main() -> None:
    args = parse_args()
    snap = Path(args.snapshot).expanduser()
    try:
        ctx = load_snapshot(snap)
        print(f"[INFO] Loaded keys: {ctx.metadata['included_keys']}")
    except Exception as e:
        print(f"[ERROR] Snapshot load failed: {e}")
        sys.exit(1)

    prompt = assemble_prompt(ctx)
    prompt_path = Path(args.output).with_name("input_tree_llm_prompt.json")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"[INFO] Prompt saved → {prompt_path}")

    if args.llm:
        output_text = llm_narrative(prompt, model=args.llm_model)
    else:
        output_text = fallback_narrative(ctx)

    out_path = Path(args.output)
    out_path.write_text(output_text, encoding="utf-8")

    if args.print:
        print("\n--- Generated Narrative ---\n")
        print(output_text)
        print("\n---------------------------")
    print(f"✅ Narrative written to {out_path}")


if __name__ == "__main__":
    main()
