# Preprocessing and data provenance

DarkPatternGuard fetches the published `dataset/dataset.tsv` from
[`yamanalab/ec-darkpattern`](https://github.com/yamanalab/ec-darkpattern), pinned
to commit `85ea9df31928d5ad6ea0b106316b441c201adbbe`. The source README says the
dark-pattern text originates from Mathur et al. (2019) and that the non-dark
text was collected from the same e-commerce sites. Its repository includes an
Apache-2.0 license. The raw JSON retains source IDs, original categories, and
the original text; it is not modified with artificial markup or whitespace.

## Input contract

Run `python scripts/fetch_princeton_raw.py` to download and map the four
in-scope categories into `data/raw/princeton_dark_patterns_raw.json`.
`src/data/clean.py` accepts JSON arrays, JSONL, or an object containing a
`records`, `data`, or `items` list. Each record must contain `pattern_type` and
one of `text`, `content`, `description`, `product_description`, or `sentence`.
The fetcher maps `Urgency`, `Scarcity`, and `Social Proof` directly and maps
`Not Dark Pattern` to `Normal`; other positive categories are excluded.

## Deterministic cleaning

For each record the pipeline:

1. normalizes Unicode with NFKC;
2. strips HTML with BeautifulSoup;
3. removes URLs;
4. lowercases and collapses whitespace;
5. drops empty records;
6. removes exact duplicate normalized text using SHA-256;
7. maps labels to `Urgency`, `Scarcity`, `Social Proof`, or `Normal`.

The selected upstream dataset provides genuine Normal examples, so no
synthetic neutral examples are added for this run. The cleaner retains
`source_id`, `source_category`, and `source_dataset` in cleaned rows.

## Splitting and leakage control

`src/data/split.py` performs an approximately 80/10/10 stratified split using
seed `1729` from `configs/base.yaml`. When `source_id` is available it uses
stratified group splitting so segments from the same source page cannot cross
partitions. Duplicate removal occurs before splitting. Validation is used for
tuning, calibration, stacking, and early stopping; the test set is consumed
only for final reporting.

Generated outputs:

- `train.csv`, `val.csv`, `test.csv`;
- `splits_manifest.json` with row and per-class counts;
- `class_weights.json` using balanced class weights;
- `data/DATA_CARD.md` with the observed distribution and provenance notes.

Run the pipeline from the repository root:

```powershell
python -m data.clean
python -m data.split
```

Use `make data` for cleaning and splitting after fetching the source.
