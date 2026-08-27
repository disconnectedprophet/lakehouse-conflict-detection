"""
One-off reproducibility check for Haiku 4.5's best config (DLS), mirroring
exactly how the Sonnet/D and Opus/L multiruns were built: run 1 reuses the
existing published result (results/haiku/summary_full.json), runs 2-5 are
fresh API calls. Writes results/multirun/haiku_DLS_summary.json.

Not part of the general --repeats flag in experiments.py (which always does
N fresh runs) because reusing an existing single-combo result as sample 1
is a one-off cost saving specific to combos that already have a completed
run, not a generally applicable default.

Run via:
  python3 scripts/multirun_haiku_DLS.py
"""

# Libraries
import json
import os
import sys
import anthropic
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments as E

# Configuration variables
MODEL_KEY = "haiku"
COMBO = "DLS"
N_RUNS = 5


def main() -> None:
    all_pairs = E.load_pairs()
    manifests = E.load_manifests()
    ddl_blocks = E.load_ddl_blocks()
    lineage = E.load_lineage()
    model_id = E.MODELS[MODEL_KEY]

    multirun_dir = os.path.join(E.RESULTS_DIR, "multirun")
    os.makedirs(multirun_dir, exist_ok=True)

    with open(os.path.join(E.RESULTS_DIR, MODEL_KEY, "summary_full.json")) as f:
        existing = json.load(f)
    row = next(r for r in existing["combinations"] if r["combo"] == COMBO)
    runs = [{
        "run": 1,
        "source": f"reused from results/{MODEL_KEY}/summary_full.json (original benchmark pass)",
        "accuracy": row["accuracy"],
        "macro_precision": row["macro_precision"],
        "macro_recall": row["macro_recall"],
        "macro_f1": row["macro_f1"],
    }]
    print(f"Run 1 (historical): acc={runs[0]['accuracy']:.4f}  macro-F1={runs[0]['macro_f1']:.4f}")

    client = anthropic.Anthropic(api_key=E.read_api_key())
    sources = set(COMBO)

    for run_idx in range(2, N_RUNS + 1):
        tag = f"{MODEL_KEY}-{COMBO}-run{run_idx}"
        ckpt = os.path.join(multirun_dir, f"ckpt_{tag}.json")
        result_file = os.path.join(multirun_dir, f"{MODEL_KEY}_{COMBO}_run{run_idx}.json")

        if os.path.exists(result_file):
            with open(result_file) as f:
                predictions = json.load(f)["predictions"]
            print(f"  [{tag}] already done — skipping API calls")
        else:
            predictions = E.run_predictions(
                client, model_id, all_pairs, sources, manifests, ddl_blocks, lineage,
                checkpoint_path=ckpt, tag=tag,
            )
            with open(result_file, "w") as f:
                json.dump({"combo": COMBO, "model": model_id, "run": run_idx, "predictions": predictions}, f, indent=2)
            if os.path.exists(ckpt):
                os.remove(ckpt)

        metrics = E.compute_metrics_for_split(all_pairs, predictions)
        row = {
            "run": run_idx,
            "source": "fresh API call",
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro"]["precision"],
            "macro_recall": metrics["macro"]["recall"],
            "macro_f1": metrics["macro"]["f1"],
        }
        runs.append(row)
        print(f"Run {run_idx}: acc={row['accuracy']:.4f}  macro-F1={row['macro_f1']:.4f}")

    def mean_std(key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    acc_m, acc_s = mean_std("accuracy")
    p_m, p_s = mean_std("macro_precision")
    r_m, r_s = mean_std("macro_recall")
    f1_m, f1_s = mean_std("macro_f1")

    summary = {
        "combo": COMBO,
        "model": model_id,
        "n_runs": N_RUNS,
        "n_pairs": len(all_pairs),
        "runs": runs,
        "accuracy_mean": round(acc_m, 4),
        "accuracy_std": round(acc_s, 4),
        "macro_precision_mean": round(p_m, 4),
        "macro_precision_std": round(p_s, 4),
        "macro_recall_mean": round(r_m, 4),
        "macro_recall_std": round(r_s, 4),
        "macro_f1_mean": round(f1_m, 4),
        "macro_f1_std": round(f1_s, 4),
    }
    out_path = os.path.join(multirun_dir, f"{MODEL_KEY}_{COMBO}_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmacro-F1 = {f1_m:.4f} ± {f1_s:.4f}  across {N_RUNS} runs")
    print(f"Written to {out_path}")

# Main guard
if __name__ == "__main__":
    main()
