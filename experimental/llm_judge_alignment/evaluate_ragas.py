import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

# Use ragas_experimental metric system
from ragas_experimental.metric.numeric import NumericMetric, numeric_metric


# -----------------------------------------------------------------------------
# Define metrics using ragas_experimental
# -----------------------------------------------------------------------------

@numeric_metric(name="answer_relevancy", range=(0.0, 1.0))
class AnswerRelevancy(NumericMetric):
    """Very simple semantic overlap: token-level F1 between answer & reference."""

    def _calc(self, reference: str, candidate: str) -> float:  # type: ignore[override]
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        common = len(set(ref_tokens) & set(cand_tokens))
        if common == 0:
            return 0.0
        precision = common / len(cand_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)


@numeric_metric(name="context_relevancy", range=(0.0, 1.0))
class ContextRelevancy(NumericMetric):
    """Placeholder context relevancy – score is 0 when no context available."""

    def _calc(self, context: str, candidate: str) -> float:  # type: ignore[override]
        if not context:
            return 0.0
        ctx_tokens = context.lower().split()
        cand_tokens = candidate.lower().split()
        common = len(set(ctx_tokens) & set(cand_tokens))
        if common == 0:
            return 0.0
        precision = common / len(cand_tokens)
        recall = common / len(ctx_tokens)
        return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def main(input_jsonl: Path, dspy_scores_csv: Path):
    print("[evaluate_ragas] Loading data…")
    records = load_jsonl(input_jsonl)
    dspy_df = pd.read_csv(dspy_scores_csv)

    metric_a = AnswerRelevancy
    metric_c = ContextRelevancy

    scores_a = []
    scores_c = []
    for rec in tqdm(records, desc="computing metrics"):
        scores_a.append(
            metric_a.score(None, reference=rec.get("reference", ""), candidate=rec["candidate"]).result
        )
        scores_c.append(
            metric_c.score(None, context=rec.get("context", ""), candidate=rec["candidate"]).result
        )

    df = pd.DataFrame({
        "answer_relevancy": scores_a,
        "context_relevancy": scores_c,
        "dspy_score": dspy_df["score"].astype(float),
    })

    for col in ["answer_relevancy", "context_relevancy"]:
        rho, p = spearmanr(df[col], df["dspy_score"])
        print(f"Spearman correlation between {col} and DSPy score: ρ={rho:.3f}, p={p:.3g}")

    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="answers.jsonl path")
    parser.add_argument("--dspy_scores", type=Path, required=True, help="CSV from evaluate_dspy.py")
    args = parser.parse_args()

    main(Path(args.input), Path(args.dspy_scores))