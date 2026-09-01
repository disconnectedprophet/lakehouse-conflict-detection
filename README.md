# Lakehouse Conflict Detection

Dataset and full factorial LLM experiment for detecting semantic integration conflicts in data lakehouse column pairs. The research study includes a non-LLM baseline and a multi-LLM comparison, together with a reproducibility check.

## Repository structure

```
.
├── dataset/            labeled column pairs, source tables, and metadata (see "Dataset")
├── results/            pre-computed evaluation results, one subdirectory per model (see "Results")
├── scripts/            one-off scripts used to generate the dataset (see "Scripts")
├── experiments.py      full factorial LLM evaluation (see "LLM experiment")
├── baseline.py         non-LLM TF-IDF + Logistic Regression baseline (see "Non-LLM baseline")
└── requirements.txt    Python dependencies
```

## Problem

When integrating tables from a data lakehouse (open table format), two classes of semantic conflict arise that are invisible to schema-level checks and to entity-level matching alike:

- TYPE1 (measure mismatch): the same concept is expressed in different units (e.g. annual revenue in USD versus EUR, venue capacity in seats versus thousands).
- TYPE2 (granularity mismatch): the same concept is aggregated at different levels (e.g. monthly versus annual sales, per-employee versus per-grade salary).

Standard column type classifiers (e.g. Sherlock) have low recall on exactly these entity types and they miss many of the relevant columns. Hence, it cannot serve as a reliable first-pass filter that groups candidate pairs for comparison. This layer therefore operates on candidate pairs directly.

## Dataset

The dataset covers the ten Sherlock entity types with the lowest per-type recall in our evaluation of the publicly released Sherlock model. More precisely, the dataset comprises ten types that fall furthest below the macro-averaged recall of 0.866 measured across all 78 types: ranking (0.312), sales (0.528), director (0.547), person (0.618), brand (0.671), nationality (0.691), gender (0.721), capacity (0.721), range (0.759), name (0.759).

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

- D: DDL (CREATE TABLE definition - declared types, column comments, and TBLPROPERTIES)
- L: Lineage (OpenLineage event with the SQL transformation that derived the table's rows from upstream tables)
- S: Statistics (row count, null count, distinct count, min/max/mean/std from manifest)
- V: Values (up to 10 sample values from CSV)

Metrics: precision, recall, F1 per class and macro-averaged.

### Prompt freeze design

The prompt and decision rules were developed on a small pilot predecessor of the benchmark and frozen before any evaluation. The model is then called once per pair on the full 190-pair dataset in a single pass. Because no prompt or rule was adjusted using these 190 pairs, the full benchmark serves as the primary untuned evaluation for each pre-specified source combination.

### Multi-size evaluation

Metrics are additionally computed on stratified subsets of 70, 119, and 190 pairs to verify that the source-combination ranking is stable across dataset sizes.

### Multi-model comparison

The identical frozen prompt and protocol is run against three models — Haiku 4.5 (`claude-haiku-4-5-20251001`), Sonnet 5 (`claude-sonnet-5`), and Opus 5 (`claude-opus-5`) — keeping model family, API, and prompt template fixed so capability tier is the only variable. This checks whether the source-ranking findings (declarative sources beating extensional ones, the full 4-source stack never ranking first, measure conflicts being the hardest class) hold as capability scales, or are specific to the originally benchmarked model (i.e. Haiku 4.5).

```
python experiments.py --model haiku    # default — writes to results/haiku/
python experiments.py --model sonnet   # writes to results/sonnet/
python experiments.py --model opus     # writes to results/opus/
python experiments.py --model all      # all three in sequence
```

### Reproducibility check

The API is called with extended thinking disabled but a non-zero default sampling temperature, so repeated calls are not deterministic. `--repeats` reruns one source combination N times against one model to estimate run-to-run variability, which the benchmark-size robustness check above should be read against:

```
python experiments.py --repeats 5 --combo DLS   # default model: haiku
```

Writes `results/multirun/<model>_<combo>_run{1..N}.json` (raw, not committed — see "What's committed" below) and an aggregated `results/multirun/<model>_<combo>_summary.json` with mean ± std, including each individual run's metrics.

### Running the experiment

```
pip install -r requirements.txt
python experiments.py
```

The API key is read from the `ANTHROPIC_API_KEY` environment variable, or from an `ANTHROPIC_API_KEY=sk-ant-...` line in `~/.zshrc` or `~/.bashrc`.

Intermediate predictions are checkpointed and support resumption if interrupted. A non-retryable API error (e.g. insufficient credit balance) aborts the run immediately with whatever was completed saved, rather than silently degrading the rest of the batch to `UNKNOWN` predictions.

## Non-LLM baseline (`baseline.py`)

TF-IDF + Logistic Regression over the same evidence text the LLM prompt is built from, evaluated on all 15 source subsets. With only 190 pairs, a single train/test split starves a 4-class model, so this uses stratified 5-fold cross-validation and reports mean ± std across folds instead. No API calls.

```
python baseline.py
```

Writes `results/baseline/summary.json`. This provides a shallow non-LLM reference point - the baseline - against which the LLM results should be read.

## Results

Pre-computed results are in `results/`, one subdirectory per model:

```
results/
    haiku/    summary_full.json, summary_sizes.json
    sonnet/   the same two files
    opus/     the same two files
    baseline/summary.json          TF-IDF + Logistic Regression, all 15 combinations, 5-fold CV
    multirun/<model>_<combo>_summary.json   5 repeats of each model's highest-scoring config, mean ± std
    multillm_comparison.json       Haiku / Sonnet / Opus side by side, all 15 combinations
```

Each `summary_*.json` reports accuracy and macro-averaged precision/recall/F1, plus per-class precision/recall/F1, for every one of the 15 evidence-source combinations. The paper reports the full-benchmark (`summary_full.json`) numbers, cross-checked for size-robustness by `summary_sizes.json`.

### What's committed

This repository keeps only **aggregated metrics** — the `summary_*.json` files above. It does not commit the raw per-pair predictions (`run_{combo}.json`, or the individual `run{k}.json` files inside `multirun/`) that those aggregates are computed from. However, `experiments.py` still writes them locally when run, they're just not part of the repository. This keeps the repository to the citable numbers rather than working files.

If you need the underlying per-pair predictions — e.g. to audit which specific pairs a model got wrong — regenerate them with `python experiments.py --model <haiku|sonnet|opus>`. This reproduces the same frozen prompt and protocol, so results land very close to what's published here (the reproducibility check above estimates run-to-run standard deviations of up to 0.0075 macro-F1), but not bit-for-bit identical, since the API samples at a non-zero default temperature.

## Scripts

The `scripts/` directory contains the generation scripts used to produce the synthetic dataset. Running them is not required to use the dataset.

- `scripts/generate_tables.py`: generates all 64 CSV tables covering numeric entity types (sales, ranking, capacity, salary, climate) and text entity types (nationality, gender, person names, brand names, director and cast names).
- `scripts/generate_pairs.py`: generates pairs.json, ddl.sql, lineage.json, and manifest JSON files from the CSV tables.
