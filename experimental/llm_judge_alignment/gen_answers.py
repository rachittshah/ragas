import argparse
import json
import os
from pathlib import Path
from typing import List

import openai
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT = DATA_DIR / "answers.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "Answer the question concisely in one or two sentences."
)


def generate_answer(question: str, model: str = "gpt-3.5-turbo-0125") -> str:
    """Query the model with minimal prompting and return the answer text."""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def main(n_examples: int, output_path: Path, overwrite: bool, model: str):
    if output_path.exists() and not overwrite:
        print(f"[gen_answers] {output_path} already exists – skipping generation.")
        return

    print("[gen_answers] Loading HotpotQA validation split via 🤗 datasets …")
    dataset = load_dataset("hotpot_qa", "distractor", split=f"validation[:{n_examples}]")

    records: List[dict] = []
    for ex in tqdm(dataset, desc="generating answers"):
        q = ex["question"]
        ref = ex["answer"]
        try:
            cand = generate_answer(q, model=model)
        except Exception as e:
            print(f"OpenAI error for question {q[:50]}…: {e}")
            cand = "<ERROR>"
        records.append({"question": q, "reference": ref, "candidate": cand})

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gen_answers] Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidate answers with GPT-3.5")
    parser.add_argument("--n", type=int, default=100, help="number of validation examples to generate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output .jsonl path")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing file")
    parser.add_argument("--model", default="gpt-3.5-turbo-0125", help="OpenAI chat model name")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    main(args.n, Path(args.output), args.overwrite, args.model)