## LLM-as-Judge Alignment Experiment

This folder contains a **reproducible experiment** that examines how well a Large-Language-Model (LLM) can act as an automatic evaluator ("judge") for open-domain question–answering.

Key objectives
---------------
1. **LLM-vs-Ragas metrics** – compare the numeric score assigned by a GPT-4-style judge (implemented with *DSPy*) against classic automatic metrics implemented in `ragas_experimental`.
2. **Correlation & calibration** – quantify Spearman/Kendall correlation and basic calibration between the two scoring families.
3. **Cost analysis** – record approximate token-level cost for scaling to 1 k datapoints.

Folder structure
----------------
```
llm_judge_alignment/
├── data/                 # downloaded dataset & cached answers
│   └── answers.jsonl
├── requirements.txt      # extra deps for this experiment
├── gen_answers.py        # step 1 – create candidate answers with GPT-3.5
├── evaluate_dspy.py      # step 2 – score answers with DSPy judge module
├── evaluate_ragas.py     # step 3 – baseline ragas_experimental metrics
└── README.md             # ← you are here
```

Quick-start
-----------
1. *Install deps* (ideally inside a venv):
   ```bash
   pip install -r requirements.txt
   ```
2. *Export your model key* (e.g. for OpenAI):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. *Generate answers* (subset of 100 HotpotQA dev examples):
   ```bash
   python gen_answers.py --n 100
   ```
4. *Run the LLM judge*:
   ```bash
   python evaluate_dspy.py --input data/answers.jsonl
   ```
5. *Compute ragas metrics + correlation plots*:
   ```bash
   python evaluate_ragas.py --input data/answers.jsonl --dspy_scores outputs/dspy_scores.csv
   ```

The final line of `evaluate_ragas.py` prints a small report with Spearman correlation between the DSPy-based judge and ragas's automatic metrics.

Notes
-----
* **Dataset** – we use the public *HotpotQA* validation split (via HuggingFace `datasets`) because it provides a ground-truth answer we can compare against.
* **Caching** – `gen_answers.py` will not re-query the LLM if `data/answers.jsonl` already exists.
* **Prompt/rubric** – the evaluation rubric lives inside `evaluate_dspy.py` in the `JudgeSignature`.  Feel free to tweak and re-run.
* **Costs** – judging 100 examples with GPT-4o-mini ≈ ¢20.  Scale accordingly.