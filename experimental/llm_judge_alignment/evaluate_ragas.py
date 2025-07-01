import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

# -----------------------------------------------------------------------------
# A *very* small numeric metric: token-level F1 between candidate & reference
# -----------------------------------------------------------------------------

def token_f1(reference: str, candidate: str) -> float:
    """Simple token-level F1 score."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    common = len(set(ref_tokens) & set(cand_tokens))
    if common == 0:
        return 0.0
    precision = common / len(cand_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def main(input_jsonl: Path, dspy_scores_csv: Path):
    print("[evaluate_ragas] Loading data…")
    records = load_jsonl(input_jsonl)
    dspy_df = pd.read_csv(dspy_scores_csv)

    ragas_scores = []
    for rec in tqdm(records, desc="ragas metric"):
        ragas_scores.append(token_f1(rec["reference"], rec["candidate"]))

    # Combine into DataFrame for correlation
    df = pd.DataFrame({
        "ragas_f1": ragas_scores,
        "dspy_score": dspy_df["score"].astype(float),
    })

    rho, p = spearmanr(df["ragas_f1"], df["dspy_score"])
    print("\n=== Correlation Report ===")
    print(f"Spearman ρ: {rho:.3f} (p={p:.3g}) over n={len(df)}")
    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="answers.jsonl path")
    parser.add_argument("--dspy_scores", type=Path, required=True, help="CSV from evaluate_dspy.py")
    args = parser.parse_args()

    main(Path(args.input), Path(args.dspy_scores))