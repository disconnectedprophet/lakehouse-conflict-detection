"""
Full factorial LLM experiment for semantic conflict detection.

Four evidence sources tested in all 15 non-empty subsets of {D, L, S, V}:
  D — DDL CREATE TABLE block with column comments and TBLPROPERTIES
  L — Lineage OpenLineage SQL query that produced the table
  S — Stats manifest: row count, nulls, distinct, min/max/mean/std
  V — Values up to 10 sample values from CSV

Prompt freeze design: the model is called once per pair on the full 190-pair
dataset with no split-awareness. Dev/test metrics are computed retroactively
from the same predictions to show that the split does not affect the result.

Multi-size evaluation: metrics are computed on stratified subsets of
SIZES pairs to show that the source-combination trend is stable across
dataset sizes.

Results written to results/:
  run_{combo}.json raw predictions for all 190 pairs
  summary_full.json metrics on all 190 pairs for all 15 combos
  summary_dev.json metrics on dev subset (~70% of 190)
  summary_test.json metrics on test subset (~30% of 190)
  summary_sizes.json metrics for each combo x size combination

Run via:
  python3 experiments.py
"""

import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from itertools import combinations as icombs

import anthropic
import pandas as pd
from sklearn.metrics import classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_FILE = os.path.join(SCRIPT_DIR, "dataset", "pairs.json")
TABLES_DIR = os.path.join(SCRIPT_DIR, "dataset", "tables")
MFST_DIR = os.path.join(SCRIPT_DIR, "dataset", "metadata", "manifests")
DDL_FILE = os.path.join(SCRIPT_DIR, "dataset", "metadata", "ddl.sql")
LINEAGE_FILE = os.path.join(SCRIPT_DIR, "dataset", "metadata", "lineage.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODEL = "claude-haiku-4-5-20251001"
MAX_SAMPLES = 10
RETRY_SLEEP = 10
MAX_RETRIES = 3
SIZES = [70, 120, 190]
SPLIT_SEED = 42
TEST_FRAC = 0.30

LABELS: list[str] = [
    "TYPE1_MEASURE",
    "TYPE2_GRANULARITY",
    "NO_CONFLICT_DUPLICATE",
    "NO_CONFLICT_DIFF_ENTITY",
]

LABEL_DESCRIPTIONS = """\
  TYPE1_MEASURE — same semantic concept, different measurement unit (e.g. USD vs EUR, seats vs thousands, raw pts vs normalised)
  TYPE2_GRANULARITY — same semantic concept, different aggregation level (e.g. daily vs monthly, per-row vs per-group aggregate)
  NO_CONFLICT_DUPLICATE — same data, no semantic conflict
  NO_CONFLICT_DIFF_ENTITY — different real-world concepts, no conflict"""


def _read_key_from_bashrc() -> str:
    with open(os.path.expanduser("~/.bashrc")) as f:
        for line in f:
            m = re.search(r'ANTHROPIC(?:_API)?_KEY=["\']?(sk-ant-[^\s"\']+)', line)
            if m:
                return m.group(1)
    raise RuntimeError("ANTHROPIC_KEY / ANTHROPIC_API_KEY not found in ~/.bashrc")


def load_pairs() -> list[dict]:
    with open(PAIRS_FILE) as f:
        return json.load(f)


def stratified_split(
    pairs: list[dict], test_frac: float = TEST_FRAC, seed: int = SPLIT_SEED
) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_label[p["label"]].append(p)
    dev: list[dict] = []
    test: list[dict] = []
    rng = random.Random(seed)
    for group in by_label.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_frac))
        test.extend(shuffled[:n_test])
        dev.extend(shuffled[n_test:])
    return dev, test


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


def call_llm(client: anthropic.Anthropic, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_prediction(msg.content[0].text)
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


def combo_label(combo: tuple) -> str:
    return "".join(sorted(combo))


def run_combo(
    combo: tuple,
    all_pairs: list[dict],
    client: anthropic.Anthropic,
    manifests: dict[str, dict],
    ddl_blocks: dict[str, str],
    lineage: dict[str, dict],
) -> dict:
    label = combo_label(combo)
    sources = set(combo)
    result_file = os.path.join(RESULTS_DIR, f"run_{label}.json")
    ckpt_file = os.path.join(RESULTS_DIR, f"ckpt_{label}.json")

    if os.path.exists(result_file):
        print(f"  [{label}] already done — skipping")
        with open(result_file) as f:
            return json.load(f)

    predictions: dict = {}
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            predictions = json.load(f)
        print(f"  [{label}] resuming from checkpoint ({len(predictions)}/{len(all_pairs)} done)")

    for i, pair in enumerate(all_pairs, 1):
        pid = pair["id"]
        if pid in predictions:
            continue
        predictions[pid] = call_llm(client, build_prompt(pair, sources, manifests, ddl_blocks, lineage))
        print(f"  [{label}] {i}/{len(all_pairs)}  {pid}: {predictions[pid]:<30} (true: {pair['label']})")
        if i % 10 == 0:
            with open(ckpt_file, "w") as f:
                json.dump(predictions, f)

    result = {"combo": label, "sources": sorted(sources), "predictions": predictions}
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
    all_results: list[dict], split_pairs: list[dict], split_name: str
) -> dict:
    rows = [
        build_summary_row(r, split_pairs)
        for r in sorted(all_results, key=lambda x: (len(x["combo"]), x["combo"]))
    ]
    summary = {"model": MODEL, "split": split_name, "n_pairs": len(split_pairs), "combinations": rows}
    with open(os.path.join(RESULTS_DIR, f"summary_{split_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_summary_sizes(all_results: list[dict], all_pairs: list[dict]) -> dict:
    rows = []
    for size in SIZES:
        subset = stratified_sample(all_pairs, size)
        for r in sorted(all_results, key=lambda x: (len(x["combo"]), x["combo"])):
            row = build_summary_row(r, subset)
            row["size"] = size
            rows.append(row)
    summary = {"model": MODEL, "sizes": SIZES, "combinations": rows}
    with open(os.path.join(RESULTS_DIR, "summary_sizes.json"), "w") as f:
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


def main() -> None:
    all_combos = [combo for r in range(1, 5) for combo in icombs(["V", "S", "D", "L"], r)]
    all_pairs = load_pairs()
    dev_pairs, test_pairs = stratified_split(all_pairs)

    print(f"Full factorial experiment — model: {MODEL}")
    print(f"  Total: {len(all_pairs)}  dev: {len(dev_pairs)}  test: {len(test_pairs)}")
    for split_name, split in [("full", all_pairs), ("dev", dev_pairs), ("test", test_pairs)]:
        dist = Counter(p["label"] for p in split)
        print(f"  {split_name}: " + "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    print(f"  multi-size subsets: {SIZES}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    client = anthropic.Anthropic(api_key=_read_key_from_bashrc())
    manifests = load_manifests()
    ddl_blocks = load_ddl_blocks()
    lineage = load_lineage()

    all_results = []
    for i, combo in enumerate(all_combos, 1):
        print(f"\n[{i}/{len(all_combos)}] Sources: {set(combo)}")
        all_results.append(run_combo(combo, all_pairs, client, manifests, ddl_blocks, lineage))

    summary_full = build_summary(all_results, all_pairs, "full")
    summary_dev = build_summary(all_results, dev_pairs, "dev")
    summary_test = build_summary(all_results, test_pairs, "test")
    build_summary_sizes(all_results, all_pairs)

    print_summary_table(summary_full, "full")
    print_summary_table(summary_dev, "dev")
    print_summary_table(summary_test, "test")
    print("\nsummary_full.json  summary_dev.json  summary_test.json  summary_sizes.json written.")


# Main guard
if __name__ == "__main__":
    main()
