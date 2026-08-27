"""
Full factorial LLM experiment for semantic conflict detection.

Four evidence sources tested in all 15 non-empty subsets of {D, L, S, V}:
  D — DDL CREATE TABLE block with column comments and TBLPROPERTIES
  L — Lineage OpenLineage SQL query that produced the table
  S — Stats manifest: row count, nulls, distinct, min/max/mean/std
  V — Values up to 10 sample values from CSV

Prompt freeze design: the model is called once per pair on the full 190-pair
dataset in a single pass, with no prompt or rule adjusted using these pairs.

Multi-size evaluation: metrics are computed on stratified subsets of
SIZES pairs to show that the source-combination trend is stable across
dataset sizes.

Multi-model evaluation (--model): the same protocol against Haiku 4.5,
Sonnet 5, or Opus 5, to check whether the source-ranking findings hold as
capability scales. Every model writes to its own results/<model>/
subdirectory (results/haiku/, results/sonnet/, results/opus/).

Reproducibility check (--repeats): repeats one source combination N times
against one model to measure run-to-run variance, since the API is called
with thinking disabled but default (non-zero) sampling temperature.

Only aggregated metrics are committed to this repository (see below); raw
per-pair predictions (run_{combo}.json, multirun run{k}.json) are working
files this script produces locally but the repo does not keep, since they
are regenerable by re-running the identical frozen prompt.

Results written to results/<model>/:
  run_{combo}.json         raw predictions for all 190 pairs (not committed)
  summary_full.json        metrics on all 190 pairs for all 15 combos
  summary_sizes.json       metrics for each combo x size combination
  multirun/<model>_<combo>_run{k}.json   raw predictions per repeat (not committed)
  multirun/<model>_<combo>_summary.json  mean/std across repeats

Run via:
  python3 experiments.py # Haiku 4.5, all 15 combos (original behavior)
  python3 experiments.py --model sonnet # Sonnet 5, all 15 combos
  python3 experiments.py --model all # Haiku + Sonnet + Opus
  python3 experiments.py --repeats 5 --combo DLS # reproducibility check on one combo
"""

# Libraries
import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations as icombs
import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_FILE = os.path.join(SCRIPT_DIR, "dataset", "pairs.json")
TABLES_DIR = os.path.join(SCRIPT_DIR, "dataset", "tables")
MFST_DIR = os.path.join(SCRIPT_DIR, "dataset", "metadata", "manifests")
DDL_FILE = os.path.join(SCRIPT_DIR, "dataset", "metadata", "ddl.sql")
LINEAGE_FILE = os.path.join(SCRIPT_DIR, "dataset", "metadata", "lineage.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}
DEFAULT_MODEL = "haiku"

MAX_SAMPLES = 10
MAX_TOKENS = 32  # covers the longest label ("NO_CONFLICT_DIFF_ENTITY") across all
# three models' tokenizers; Haiku needs <16, Sonnet/Opus need up to 22.
MAX_RETRIES = 3
RETRY_SLEEP = 10
MAX_WORKERS = 8
SIZES = [70, 120, 190]
SPLIT_SEED = 42

LABELS: list[str] = [
    "TYPE1_MEASURE",
    "TYPE2_GRANULARITY",
    "NO_CONFLICT_DUPLICATE",
    "NO_CONFLICT_DIFF_ENTITY",
]

LABEL_DESCRIPTIONS = """\
  TYPE1_MEASURE - same semantic concept, different measurement unit (e.g. USD vs EUR, seats vs thousands, raw pts vs normalised)
  TYPE2_GRANULARITY - same semantic concept, different aggregation level (e.g. daily vs monthly, per-row vs per-group aggregate)
  NO_CONFLICT_DUPLICATE — same data, no semantic conflict
  NO_CONFLICT_DIFF_ENTITY — different real-world concepts, no conflict"""


def read_api_key() -> str:
    """Look for the Anthropic API key: env vars first, then shell rc files
    (some setups use ~/.bashrc, others ~/.zshrc depending on the shell)."""
    val = os.environ.get("ANTHROPIC_API_KEY")
    if val:
        return val
    for rc in ("~/.zshrc", "~/.bashrc", "~/.zshenv", "~/.profile"):
        path = os.path.expanduser(rc)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = re.search(r'ANTHROPIC(?:_API)?_KEY=["\']?(sk-ant-[^\s"\']+)', line)
                if m:
                    return m.group(1)
    raise RuntimeError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY in the environment, "
        "or add an ANTHROPIC_API_KEY=sk-ant-... line to ~/.zshrc or ~/.bashrc."
    )


def load_pairs() -> list[dict]:
    with open(PAIRS_FILE) as f:
        return json.load(f)


def stratified_sample(pairs: list[dict], n: int, seed: int = SPLIT_SEED) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_label[p["label"]].append(p)
    total = len(pairs)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for label, group in by_label.items():
        k = max(1, round(len(group) * n / total))
        shuffled = group[:]
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:k])
    rng.shuffle(sampled)
    return sampled[:n]


def load_manifests() -> dict[str, dict]:
    mfsts: dict[str, dict] = {}
    for fname in os.listdir(MFST_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(MFST_DIR, fname)) as f:
                m = json.load(f)
            mfsts[m["table"]] = m["columns"]
    return mfsts


def load_ddl_blocks() -> dict[str, str]:
    with open(DDL_FILE) as f:
        text = f.read()
    blocks: dict[str, str] = {}
    for part in re.split(r'(?=CREATE TABLE )', text):
        m = re.search(r'CREATE TABLE (\w+)', part)
        if m:
            blocks[m.group(1)] = part.strip()
    return blocks


def load_lineage() -> dict[str, dict]:
    with open(LINEAGE_FILE) as f:
        events = json.load(f)
    by_table: dict[str, dict] = {}
    for e in events:
        for o in e.get("outputs", []):
            name = o.get("name")
            if name:
                by_table[name] = {
                    "job": e.get("job", {}).get("name", ""),
                    "inputs": [i["name"] for i in e.get("inputs", [])],
                    "sql": o.get("facets", {}).get("sql", {}).get("query", ""),
                }
    return by_table


def get_sample_values(table: str, col: str, n: int = MAX_SAMPLES) -> list[str]:
    try:
        df = pd.read_csv(os.path.join(TABLES_DIR, f"{table}.csv"), usecols=[col], low_memory=False)
        vals = df[col].dropna().tolist()
        random.seed(42)
        return [str(v) for v in random.sample(vals, min(n, len(vals)))]
    except Exception:
        return []


def block_V(table: str, col: str) -> str:
    vals = get_sample_values(table, col)
    vals_str = ", ".join(f'"{v}"' for v in vals) if vals else "(no data)"
    return f"  {table}.{col} — sample values: [{vals_str}]"


def block_S(table: str, col: str, manifests: dict[str, dict]) -> str:
    info = manifests.get(table, {}).get(col, {})
    if not info:
        return f"  {table}.{col} — no statistics available"
    parts = [
        f"rows={info.get('row_count', '?')}",
        f"nulls={info.get('null_count', '?')}",
        f"distinct={info.get('distinct_count', '?')}",
    ]
    if "min" in info:
        parts += [f"min={info['min']}", f"max={info['max']}",
                  f"mean={info['mean']}", f"std={info['std']}"]
    return f"  {table}.{col} — {', '.join(parts)}"


def block_D(table: str, ddl_blocks: dict[str, str]) -> str:
    return ddl_blocks.get(table, f"(DDL not available for {table})")


def block_L(table: str, lineage: dict[str, dict]) -> str:
    entry = lineage.get(table)
    if not entry:
        return f"  {table} — no lineage available"
    inputs = ", ".join(entry["inputs"]) if entry["inputs"] else "none"
    return "\n".join([
        f"  table: {table}",
        f"  job: {entry['job']}",
        f"  inputs: {inputs}",
        f"  sql: {entry['sql']}",
    ])


def build_prompt(
    pair: dict,
    sources: set[str],
    manifests: dict[str, dict],
    ddl_blocks: dict[str, str],
    lineage: dict[str, dict],
) -> str:
    ta, ca = pair["table_a"], pair["col_a"]
    tb, cb = pair["table_b"], pair["col_b"]
    sections = [
        "You are classifying a column pair from a data lakehouse.\n",
        f"Column pair:\n  A: {ta}.{ca}\n  B: {tb}.{cb}\n",
    ]
    if "V" in sources:
        sections.append("Sample values:\n" + block_V(ta, ca) + "\n" + block_V(tb, cb))
    if "S" in sources:
        sections.append("Column statistics:\n" + block_S(ta, ca, manifests) + "\n" + block_S(tb, cb, manifests))
    if "D" in sources:
        sections.append(
            "Schema (DDL):\n--- Table A ---\n" + block_D(ta, ddl_blocks)
            + "\n--- Table B ---\n" + block_D(tb, ddl_blocks)
        )
    if "L" in sources:
        sections.append(
            "Lineage:\n--- Table A ---\n" + block_L(ta, lineage)
            + "\n--- Table B ---\n" + block_L(tb, lineage)
        )
    sections.append(
        "Classes:\n" + LABEL_DESCRIPTIONS + "\n\n"
        "Respond with exactly one label (no explanation):\n"
        "TYPE1_MEASURE | TYPE2_GRANULARITY | NO_CONFLICT_DUPLICATE | NO_CONFLICT_DIFF_ENTITY"
    )
    return "\n\n".join(sections)


def parse_prediction(text: str) -> str:
    text = text.strip()
    for label in LABELS:
        if label in text:
            return label
    upper = text.upper().replace(" ", "_")
    for label in LABELS:
        if label in upper:
            return label
    return "UNKNOWN"


def call_llm(client: anthropic.Anthropic, model_id: str, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            if not text_blocks:
                raise ValueError(f"no text block in response content: {[type(b).__name__ for b in msg.content]}")
            return parse_prediction("".join(text_blocks))
        except anthropic.BadRequestError as e:
            # 400-class errors (invalid request, insufficient credit balance, etc.)
            # won't be fixed by retrying identical requests. Abort immediately
            # instead of quietly degrading every remaining pair to "UNKNOWN".
            raise RuntimeError(f"Fatal (non-retryable) API error, aborting: {e}") from e
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP)
    return "UNKNOWN"


def compute_metrics(true_labels: list[str], pred_labels: list[str]) -> dict:
    report = classification_report(
        true_labels, pred_labels, labels=LABELS, output_dict=True, zero_division=0
    )
    result: dict = {}
    for label in LABELS:
        r = report.get(label, {})
        result[label] = {
            "precision": round(r.get("precision", 0), 4),
            "recall": round(r.get("recall", 0), 4),
            "f1": round(r.get("f1-score", 0), 4),
            "support": int(r.get("support", 0)),
        }
    macro = report.get("macro avg", {})
    result["macro"] = {
        "precision": round(macro.get("precision", 0), 4),
        "recall": round(macro.get("recall", 0), 4),
        "f1": round(macro.get("f1-score", 0), 4),
    }
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    result["accuracy"] = round(correct / len(true_labels), 4)
    return result


def compute_metrics_for_split(pairs: list[dict], predictions: dict) -> dict:
    return compute_metrics(
        [p["label"] for p in pairs],
        [predictions.get(p["id"], "UNKNOWN") for p in pairs],
    )


def combo_label(combo) -> str:
    return "".join(sorted(combo))


def results_dir_for(model_key: str) -> str:
    return os.path.join(RESULTS_DIR, model_key)


def run_predictions(
    client: anthropic.Anthropic,
    model_id: str,
    pairs: list[dict],
    sources: set[str],
    manifests: dict[str, dict],
    ddl_blocks: dict[str, str],
    lineage: dict[str, dict],
    checkpoint_path: str,
    tag: str,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Classify every pair under one model + one source subset. Checkpointed
    (resumable) and lightly concurrent to keep wall time down; aborts and
    saves partial progress immediately on a fatal (non-retryable) error
    instead of burning through the rest of the batch's retries."""
    predictions: dict = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            predictions = json.load(f)
        print(f"  [{tag}] resuming from checkpoint ({len(predictions)}/{len(pairs)} done)")

    todo = [p for p in pairs if p["id"] not in predictions]
    if not todo:
        return predictions

    def work(pair):
        prompt = build_prompt(pair, sources, manifests, ddl_blocks, lineage)
        return pair["id"], call_llm(client, model_id, prompt), pair["label"]

    done_count = len(predictions)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, p) for p in todo]
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                pid, pred, true_label = fut.result()
                predictions[pid] = pred
                done_count += 1
                print(f"  [{tag}] {done_count}/{len(pairs)}  {pid}: {pred:<28} (true: {true_label})")
                if i % 20 == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump(predictions, f)
        except Exception:
            for f in futures:
                f.cancel()
            with open(checkpoint_path, "w") as f:
                json.dump(predictions, f)
            print(f"  [{tag}] aborted — {done_count}/{len(pairs)} saved to checkpoint")
            raise

    with open(checkpoint_path, "w") as f:
        json.dump(predictions, f)
    return predictions


def run_combo(
    combo,
    all_pairs: list[dict],
    client: anthropic.Anthropic,
    model_key: str,
    model_id: str,
    manifests: dict[str, dict],
    ddl_blocks: dict[str, str],
    lineage: dict[str, dict],
    results_dir: str,
) -> dict:
    label = combo_label(combo)
    sources = set(combo)
    result_file = os.path.join(results_dir, f"run_{label}.json")
    ckpt_file = os.path.join(results_dir, f"ckpt_{label}.json")

    if os.path.exists(result_file):
        print(f"  [{label}] already done — skipping")
        with open(result_file) as f:
            return json.load(f)

    predictions = run_predictions(
        client, model_id, all_pairs, sources, manifests, ddl_blocks, lineage,
        checkpoint_path=ckpt_file, tag=f"{model_key}-{label}",
    )

    result = {"combo": label, "sources": sorted(sources), "model": model_id, "predictions": predictions}
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
    return result


def build_summary_row(r: dict, split_pairs: list[dict]) -> dict:
    metrics = compute_metrics_for_split(split_pairs, r["predictions"])
    row: dict = {
        "combo": r["combo"],
        "sources": r["sources"],
        "n_pairs": len(split_pairs),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro"]["precision"],
        "macro_recall": metrics["macro"]["recall"],
        "macro_f1": metrics["macro"]["f1"],
    }
    for label in LABELS:
        row[f"{label}_p"] = metrics[label]["precision"]
        row[f"{label}_r"] = metrics[label]["recall"]
        row[f"{label}_f1"] = metrics[label]["f1"]
    return row


def build_summary(
    all_results: list[dict], split_pairs: list[dict], split_name: str, model_id: str, results_dir: str
) -> dict:
    rows = [
        build_summary_row(r, split_pairs)
        for r in sorted(all_results, key=lambda x: (len(x["combo"]), x["combo"]))
    ]
    summary = {"model": model_id, "split": split_name, "n_pairs": len(split_pairs), "combinations": rows}
    with open(os.path.join(results_dir, f"summary_{split_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_summary_sizes(all_results: list[dict], all_pairs: list[dict], model_id: str, results_dir: str) -> dict:
    rows = []
    for size in SIZES:
        subset = stratified_sample(all_pairs, size)
        for r in sorted(all_results, key=lambda x: (len(x["combo"]), x["combo"])):
            row = build_summary_row(r, subset)
            row["size"] = size
            rows.append(row)
    summary = {"model": model_id, "sizes": SIZES, "combinations": rows}
    with open(os.path.join(results_dir, "summary_sizes.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary_table(summary: dict, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"\n{'Combo':<8} {'N':>5} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6}{tag}")
    print("-" * 42)
    for row in summary["combinations"]:
        print(
            f"{row['combo']:<8} {row['n_pairs']:>5} {row['accuracy']:>6.4f} "
            f"{row['macro_precision']:>6.4f} "
            f"{row['macro_recall']:>6.4f} "
            f"{row['macro_f1']:>6.4f}"
        )


def run_model(model_key: str, all_pairs: list[dict], client: anthropic.Anthropic,
              manifests: dict, ddl_blocks: dict, lineage: dict) -> None:
    model_id = MODELS[model_key]
    results_dir = results_dir_for(model_key)
    os.makedirs(results_dir, exist_ok=True)

    all_combos = [combo for r in range(1, 5) for combo in icombs(["V", "S", "D", "L"], r)]

    print(f"\n=== {model_key} ({model_id}) — full factorial, 15 combos ===")
    all_results = []
    for i, combo in enumerate(all_combos, 1):
        print(f"[{i}/{len(all_combos)}] Sources: {set(combo)}")
        all_results.append(run_combo(combo, all_pairs, client, model_key, model_id, manifests, ddl_blocks, lineage, results_dir))

    summary_full = build_summary(all_results, all_pairs, "full", model_id, results_dir)
    build_summary_sizes(all_results, all_pairs, model_id, results_dir)

    print_summary_table(summary_full, f"{model_key} full")
    print(f"\n{model_key}: summary_{{full,sizes}}.json written to {results_dir}/")


def run_repeats(model_key: str, combo_str: str, n_repeats: int, all_pairs: list[dict],
                 client: anthropic.Anthropic, manifests: dict, ddl_blocks: dict, lineage: dict) -> None:
    """Reproducibility check: repeat one source combination N times against
    one model and report mean/std, since the API samples at a non-zero
    default temperature."""
    model_id = MODELS[model_key]
    sources = set(combo_str.upper())
    label = combo_label(sources)
    multirun_dir = os.path.join(RESULTS_DIR, "multirun")
    os.makedirs(multirun_dir, exist_ok=True)

    runs = []
    for k in range(1, n_repeats + 1):
        tag = f"{model_key}-{label}-run{k}"
        result_file = os.path.join(multirun_dir, f"{model_key}_{label}_run{k}.json")
        ckpt_file = os.path.join(multirun_dir, f"ckpt_{tag}.json")

        if os.path.exists(result_file):
            with open(result_file) as f:
                predictions = json.load(f)["predictions"]
            print(f"[{tag}] already done — skipping API calls")
        else:
            predictions = run_predictions(
                client, model_id, all_pairs, sources, manifests, ddl_blocks, lineage,
                checkpoint_path=ckpt_file, tag=tag,
            )
            with open(result_file, "w") as f:
                json.dump({"combo": label, "model": model_id, "run": k, "predictions": predictions}, f, indent=2)
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)

        metrics = compute_metrics_for_split(all_pairs, predictions)
        row = {
            "run": k,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro"]["precision"],
            "macro_recall": metrics["macro"]["recall"],
            "macro_f1": metrics["macro"]["f1"],
        }
        runs.append(row)
        print(f"[{tag}] acc={row['accuracy']:.4f}  macro-F1={row['macro_f1']:.4f}")

    def mean_std(key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    acc_m, acc_s = mean_std("accuracy")
    p_m, p_s = mean_std("macro_precision")
    r_m, r_s = mean_std("macro_recall")
    f1_m, f1_s = mean_std("macro_f1")

    summary = {
        "combo": label, "model": model_id, "n_runs": n_repeats, "n_pairs": len(all_pairs), "runs": runs,
        "accuracy_mean": round(acc_m, 4), "accuracy_std": round(acc_s, 4),
        "macro_precision_mean": round(p_m, 4), "macro_precision_std": round(p_s, 4),
        "macro_recall_mean": round(r_m, 4), "macro_recall_std": round(r_s, 4),
        "macro_f1_mean": round(f1_m, 4), "macro_f1_std": round(f1_s, 4),
    }
    out_path = os.path.join(multirun_dir, f"{model_key}_{label}_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nmacro-F1 = {f1_m:.4f} ± {f1_s:.4f}  across {n_repeats} runs of {model_key}/{label}")
    print(f"Written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default=DEFAULT_MODEL,
                         help="which model to evaluate (default: haiku, matching the published benchmark)")
    parser.add_argument("--repeats", type=int, default=1,
                         help="repeat --combo N times against --model to measure run-to-run variance")
    parser.add_argument("--combo", type=str, default="DLS",
                         help="source combo to repeat when --repeats > 1, e.g. 'DLS' (default: DLS, the published best config)")
    args = parser.parse_args()

    all_pairs = load_pairs()

    dist = Counter(p["label"] for p in all_pairs)
    print(f"Dataset: {len(all_pairs)} pairs  " + "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))

    client = anthropic.Anthropic(api_key=read_api_key())
    manifests = load_manifests()
    ddl_blocks = load_ddl_blocks()
    lineage = load_lineage()

    if args.repeats > 1:
        run_repeats(args.model if args.model != "all" else DEFAULT_MODEL, args.combo, args.repeats,
                    all_pairs, client, manifests, ddl_blocks, lineage)
        return

    models_to_run = list(MODELS) if args.model == "all" else [args.model]
    for model_key in models_to_run:
        run_model(model_key, all_pairs, client, manifests, ddl_blocks, lineage)

# Main guard
if __name__ == "__main__":
    main()
