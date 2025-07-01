import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ragas_experimental.metric.numeric import NumericMetric, numeric_metric
from scipy.stats import spearmanr
from tqdm import tqdm

# -----------------------------------------------------------------------------
# A *very* small numeric metric: token-level F1 between candidate & reference
# -----------------------------------------------------------------------------

@numeric_metric(name="token_f1", range=(0.0, 1.0))
class TokenF1(NumericMetric):
    """Compute token-level F1 between candidate and reference answers."""

    def _calc(self, reference: str, candidate: str) -> float:  # type: ignore[override]
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        common = len(set(ref_tokens) & set(cand_tokens))
        if common == 0:
            return 0.0
        precision = common / len(cand_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    # ragas_experimental expects `run` method already via base class; we just need
    # a wrapper for a single example.
    def run_single(self, reference: str, candidate: str) -> float:
        return self._calc(reference, candidate)


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def main(input_jsonl: Path, dspy_scores_csv: Path):
    print("[evaluate_ragas] Loading data…")
    records = load_jsonl(input_jsonl)
    dspy_df = pd.read_csv(dspy_scores_csv)

    f1_metric = TokenF1()

    ragas_scores = []
    for rec in tqdm(records, desc="ragas metric"):
        ragas_scores.append(f1_metric.run_single(rec["reference"], rec["candidate"]))

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