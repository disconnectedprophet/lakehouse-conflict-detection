# Lakehouse Conflict Detection

Dataset and full factorial LLM experiment for detecting semantic integration conflicts in data lakehouse column pairs.

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

## Experiment

Full factorial evaluation across all 15 non-empty subsets of four evidence sources:

- D: DDL (CREATE TABLE block with column comments and TBLPROPERTIES)
- L: Lineage (OpenLineage SQL query that produced the table)
- S: Statistics (row count, null count, distinct count, min/max/mean/std from manifest)
- V: Values (up to 10 sample values from CSV)

Model: claude-haiku-4-5-20251001. Metrics: precision, recall, F1 per class and macro-averaged.

### Prompt freeze design

The model is called once per pair on the full 190-pair dataset with no split-awareness. Dev and test metrics are computed retroactively from the same predictions, demonstrating that the split does not affect results (no prompt optimization was performed on the dev set).

### Multi-size evaluation

Metrics are additionally computed on stratified subsets of 70, 119, and 190 pairs to verify that the source-combination ranking is stable across dataset sizes.

### Running the experiment

```
pip install -r requirements.txt
python experiments.py
```

The API key is read from `ANTHROPIC_KEY` or `ANTHROPIC_API_KEY` in the author's `~/.bashrc`.

Intermediate predictions are saved to `results/run_{combo}.json` after each combination and support resumption if interrupted.

## Results

Pre-computed results are in `results/`:

- `summary_full.json` - metrics on all 190 pairs for all 15 combinations
- `summary_dev.json` - metrics on dev subset (~133 pairs)
- `summary_test.json` - metrics on test subset (~57 pairs)
- `summary_sizes.json` - metrics for each combination across subset sizes 70/119/190

## Scripts

The `scripts/` directory contains the generation scripts used to produce the synthetic dataset. Running them is not required to use the dataset.

- `scripts/generate_tables.py`: generates all 64 CSV tables covering numeric entity types (sales, ranking, capacity, salary, climate) and text entity types (nationality, gender, person names, brand names, director and cast names).
- `scripts/generate_pairs.py`: generates pairs.json, ddl.sql, lineage.json, and manifest JSON files from the CSV tables.
