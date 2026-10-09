# Preprocessing and data provenance

DarkPatternGuard uses the Princeton University e-commerce dark-pattern dataset
provided for the course project. The source export is placed manually in
`data/raw/`; the repository never downloads a Kaggle or Hugging Face dataset.
The raw files are intentionally kept separate from generated cleaned data.

## Input contract

`src/data/clean.py` accepts JSON arrays, JSONL, or an object containing a
`records`, `data`, or `items` list. Each record must contain `pattern_type` and
one of `text`, `content`, `description`, `product_description`, or `sentence`.

## Deterministic cleaning

For each record the pipeline:

1. normalizes Unicode with NFKC;
2. strips HTML with BeautifulSoup;
3. removes URLs;
4. lowercases and collapses whitespace;
5. drops empty records;
6. removes exact duplicate normalized text using SHA-256;
7. maps labels to `Urgency`, `Scarcity`, `Social Proof`, or `Normal`.

If the neutral class is absent or below 25% of the non-neutral count,
deterministic neutral product-description examples are injected. These records
are marked `is_synthetic=true` so their provenance remains visible.

## Splitting and leakage control

`src/data/split.py` performs an 80/10/10 stratified split using seed `1729`
from `configs/base.yaml` and writes the split manifest. Duplicate removal occurs
before splitting, and tests assert that normalized text does not overlap across
the three partitions. Validation is used for tuning, calibration, stacking, and
early stopping. The test set is consumed only for final reporting.

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

Use `make data` as the equivalent project command.
