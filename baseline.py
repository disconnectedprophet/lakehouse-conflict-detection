"""
Non-LLM baseline: TF-IDF + Logistic Regression over the same evidence text
the LLM prompt in experiments.py is built from, evaluated on all 15
non-empty source subsets of {V, S, D, L} so it lines up with the main
factorial results.

With only 190 labeled pairs, a single train/test split leaves too little
training data for a 4-class problem, so this uses stratified 5-fold CV
instead and reports mean ± std across folds, plus a pooled out-of-fold
classification report per combo.

No API calls - pure sklearn.

Run via:
  python3 baseline.py
"""

# Libraries
import json
import os
from itertools import combinations as icombs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from experiments import (
    RESULTS_DIR,
    block_D,
    block_L,
    block_S,
    block_V,
    combo_label,
    compute_metrics,
    load_ddl_blocks,
    load_lineage,
    load_manifests,
    load_pairs,
)

# Configuration variables
N_FOLDS = 5
SPLIT_SEED = 42
BASELINE_RESULTS_DIR = os.path.join(RESULTS_DIR, "baseline")


def build_evidence_text(pair: dict, sources: set, manifests: dict, ddl_blocks: dict, lineage: dict) -> str:
    """Concatenate the requested evidence blocks into plain text, using the
    same block builders as the LLM prompt, so the baseline sees exactly the
    same information — just without the LLM."""
    ta, ca = pair["table_a"], pair["col_a"]
    tb, cb = pair["table_b"], pair["col_b"]
    parts = [f"A: {ta}.{ca}", f"B: {tb}.{cb}"]
    if "V" in sources:
        parts.append(block_V(ta, ca))
        parts.append(block_V(tb, cb))
    if "S" in sources:
        parts.append(block_S(ta, ca, manifests))
        parts.append(block_S(tb, cb, manifests))
    if "D" in sources:
        parts.append(block_D(ta, ddl_blocks))
        parts.append(block_D(tb, ddl_blocks))
    if "L" in sources:
        parts.append(block_L(ta, lineage))
        parts.append(block_L(tb, lineage))
    return "\n".join(parts)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SPLIT_SEED)),
    ])


def evaluate_combo(texts: list[str], labels: list[str]) -> dict:
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)

    fold_metrics = []
    oof_preds = [None] * len(texts)
    for train_idx, test_idx in skf.split(texts, labels_arr):
        pipe = make_pipeline()
        pipe.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        preds = pipe.predict([texts[i] for i in test_idx])
        for idx, p in zip(test_idx, preds):
            oof_preds[idx] = p
        fold_metrics.append(compute_metrics([labels[i] for i in test_idx], list(preds)))

    def agg(*path):
        vals = [_get(m, path) for m in fold_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    def _get(m, path):
        cur = m
        for k in path:
            cur = cur[k]
        return cur

    acc_mean, acc_std = agg("accuracy")
    p_mean, p_std = agg("macro", "precision")
    r_mean, r_std = agg("macro", "recall")
    f1_mean, f1_std = agg("macro", "f1")

    return {
        "n_pairs": len(texts),
        "n_folds": N_FOLDS,
        "accuracy_mean": round(acc_mean, 4),
        "accuracy_std": round(acc_std, 4),
        "macro_precision_mean": round(p_mean, 4),
        "macro_precision_std": round(p_std, 4),
        "macro_recall_mean": round(r_mean, 4),
        "macro_recall_std": round(r_std, 4),
        "macro_f1_mean": round(f1_mean, 4),
        "macro_f1_std": round(f1_std, 4),
        "pooled_out_of_fold": compute_metrics(labels, oof_preds),
    }


def main() -> None:
    os.makedirs(BASELINE_RESULTS_DIR, exist_ok=True)
    all_pairs = load_pairs()
    manifests = load_manifests()
    ddl_blocks = load_ddl_blocks()
    lineage = load_lineage()
    labels = [p["label"] for p in all_pairs]

    all_combos = [combo for r in range(1, 5) for combo in icombs(["V", "S", "D", "L"], r)]

    rows = []
    for combo in sorted(all_combos, key=lambda c: (len(c), combo_label(c))):
        sources = set(combo)
        label = combo_label(combo)
        texts = [build_evidence_text(p, sources, manifests, ddl_blocks, lineage) for p in all_pairs]
        metrics = evaluate_combo(texts, labels)
        row = {"combo": label, "sources": sorted(sources), **metrics}
        rows.append(row)
        print(
            f"  [{label:<4}] acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
            f"macro-F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}"
        )

    summary = {"model": "tfidf+logreg", "n_folds": N_FOLDS, "n_pairs": len(all_pairs), "combinations": rows}
    out_path = os.path.join(BASELINE_RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWritten to {out_path}")

# Main guard
if __name__ == "__main__":
    main()
