# Lakehouse Conflict Detection

Dataset and full factorial LLM experiment for detecting semantic integration conflicts in data lakehouse column pairs, with a non-LLM baseline, a multi-model comparison, and a reproducibility check.

## Problem

When integrating tables from a data lakehouse, two classes of semantic conflict arise that are invisible to schema-level checks:

- TYPE1 (measure mismatch): the same concept is expressed in different units (e.g. annual revenue in USD vs EUR, venue capacity in seats vs thousands).
- TYPE2 (granularity mismatch): the same concept is aggregated at different levels (e.g. monthly vs annual sales, per-employee vs per-grade salary).

Standard column type classifiers (e.g. Sherlock) have low recall on the entity types where these conflicts are most common, making them unsuitable as a first-pass filter. This repository targets those entity types directly.

## Dataset

The dataset covers 10 Sherlock entity types with recall below 0.75: ranking (0.312), sales (0.528), director (0.547), person (0.618), brand (0.671), nationality (0.691), gender (0.721), capacity (0.721), range (0.759), name (0.759).

```
dataset/
    pairs.json              190 labeled column pairs
    tables/                 64 CSV tables
    metadata/
        ddl.sql             CREATE TABLE definitions for all 64 tables
        lineage.json        OpenLineage events for all 64 tables
        manifests/          per-table column statistics (64 JSON files)
```

### Label distribution

| Label | Count | % |
|---|---|---|
| TYPE1_MEASURE | 59 | 31.1 |
| TYPE2_GRANULARITY | 37 | 19.5 |
| NO_CONFLICT_DUPLICATE | 60 | 31.6 |
| NO_CONFLICT_DIFF_ENTITY | 34 | 17.9 |

## LLM experiment (`experiments.py`)

Full factorial evaluation across all 15 non-empty subsets of four evidence sources:

- D: DDL (CREATE TABLE block with column comments and TBLPROPERTIES)
- L: Lineage (OpenLineage SQL query that produced the table)
- S: Statistics (row count, null count, distinct count, min/max/mean/std from manifest)
- V: Values (up to 10 sample values from CSV)

Metrics: precision, recall, F1 per class and macro-averaged.

### Prompt freeze design

The model is called once per pair on the full 190-pair dataset with no split-awareness. Dev and test metrics are computed retroactively from the same predictions, demonstrating that the split does not affect results (no prompt optimization was performed on the dev set).

### Multi-size evaluation

Metrics are additionally computed on stratified subsets of 70, 119, and 190 pairs to verify that the source-combination ranking is stable across dataset sizes.

### Multi-model comparison

The identical frozen prompt and protocol is run against three models — Haiku 4.5 (`claude-haiku-4-5-20251001`), Sonnet 5 (`claude-sonnet-5`), and Opus 5 (`claude-opus-5`) — keeping model family, API, and prompt template fixed so capability tier is the only variable. This checks whether the source-ranking findings (declarative sources beating extensional ones, the full 4-source stack never being optimal, measure conflicts being the hardest class) hold as capability scales, or are specific to the originally benchmarked model.

```
python experiments.py --model haiku    # default — writes to results/haiku/
python experiments.py --model sonnet   # writes to results/sonnet/
python experiments.py --model opus     # writes to results/opus/
python experiments.py --model all      # all three in sequence
```

### Reproducibility check

The API is called with extended thinking disabled but a non-zero default sampling temperature, so repeated calls are not deterministic. `--repeats` reruns one source combination N times against one model to establish the run-to-run noise floor, which the split-invariance and benchmark-size robustness checks above should be read against:

```
python experiments.py --repeats 5 --combo DLS   # default model: haiku
```

Writes `results/multirun/<model>_<combo>_run{1..N}.json` (raw, not committed — see "What's committed" below) and an aggregated `results/multirun/<model>_<combo>_summary.json` with mean ± std, including each individual run's metrics.

### Running the experiment

```
pip install -r requirements.txt
python experiments.py
```

The API key is read from the `ANTHROPIC_API_KEY` or `ANTHROPIC_KEY` environment variable, or from an `ANTHROPIC_API_KEY=sk-ant-...` line in `~/.zshrc` or `~/.bashrc`.

Intermediate predictions are checkpointed and support resumption if interrupted; a non-retryable API error (e.g. insufficient credit balance) aborts the run immediately with whatever was completed saved, rather than silently degrading the rest of the batch to `UNKNOWN` predictions.

## Non-LLM baseline (`baseline.py`)

TF-IDF + Logistic Regression over the same evidence text the LLM prompt is built from, evaluated on all 15 source subsets. With only 190 pairs, a single train/test split starves a 4-class model, so this uses stratified 5-fold cross-validation and reports mean ± std across folds instead. No API calls.

```
python baseline.py
```

Writes `results/baseline/summary.json`. This establishes how much of the task a bag-of-words model can already solve without an LLM, as a lower bound the LLM results should be read against.

## Results

Pre-computed results are in `results/`, one subdirectory per model:

```
results/
    haiku/    summary_full.json, summary_dev.json, summary_test.json, summary_sizes.json
    sonnet/   the same four files
    opus/     the same four files
    baseline/summary.json          TF-IDF + Logistic Regression, all 15 combinations, 5-fold CV
    multirun/haiku_DLS_summary.json   5 repeats of Haiku's best config, mean ± std
    multillm_comparison.json       Haiku / Sonnet / Opus side by side, all 15 combinations
```

Each `summary_*.json` reports accuracy and macro-averaged precision/recall/F1, plus per-class precision/recall/F1, for every one of the 15 evidence-source combinations.

### What's committed

This repository keeps only **aggregated metrics** — the `summary_*.json` files above. It does not commit the raw per-pair predictions (`run_{combo}.json`, or the individual `run{k}.json` files inside `multirun/`) that those aggregates are computed from; `experiments.py` still writes them locally when run, they're just not part of the repo. This keeps the repository to the citable numbers rather than working files.

If you need the underlying per-pair predictions — e.g. to audit which specific pairs a model got wrong — regenerate them with `python experiments.py --model <haiku|sonnet|opus>`. This reproduces the same frozen prompt and protocol, so results land very close to what's published here (the reproducibility check above puts the run-to-run noise floor at ±0.0075 macro-F1), but not bit-for-bit identical, since the API samples at a non-zero default temperature.

## Scripts

The `scripts/` directory contains the generation scripts used to produce the synthetic dataset. Running them is not required to use the dataset.

- `scripts/generate_tables.py`: generates all 64 CSV tables covering numeric entity types (sales, ranking, capacity, salary, climate) and text entity types (nationality, gender, person names, brand names, director and cast names).
- `scripts/generate_pairs.py`: generates pairs.json, ddl.sql, lineage.json, and manifest JSON files from the CSV tables.
