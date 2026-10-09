# DarkPatternGuard data card

## Summary

This project uses the Princeton University e-commerce dark-pattern dataset
provided for the NLP course. Raw exports belong in `data/raw/` and are not
downloaded by the repository. The cleaned dataset is produced by
`src/data/clean.py` and split by `src/data/split.py`.

The card is a tracked provenance template when no local raw data is present.
After running `make data`, replace the generated counts below with the
splitter's observed counts and retain the generated manifest as the
authoritative machine-readable record.

## Intended use

The data supports research and classroom evaluation of four-way e-commerce text
classification: `Urgency`, `Scarcity`, `Social Proof`, and `Normal`. It is not
intended as a consumer-facing claim that any particular phrase is legally
deceptive.

## Processing

Unicode NFKC normalization, HTML removal, URL removal, lowercasing, whitespace
normalization, empty-row filtering, and SHA-256 exact deduplication are applied
before a deterministic stratified 80/10/10 split. Labels are normalized to the
four canonical names. Neutral examples are injected only when `Normal` is
missing or below the 25% balancing threshold; injected rows are marked
`is_synthetic=true`.

## Distribution

| Partition | Rows | Urgency | Scarcity | Social Proof | Normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | generated locally | generated locally | generated locally | generated locally | generated locally |
| Validation | generated locally | generated locally | generated locally | generated locally | generated locally |
| Test | generated locally | generated locally | generated locally | generated locally | generated locally |

Run `make data` to populate the authoritative counts in
`data/cleaned/splits_manifest.json`. The fixed seeds are `1729`, `2718`, and
`3141` for model runs, with `1729` as the split seed.

## Limitations and risks

Scraped e-commerce language can reflect retailer, category, language, and time
period bias. Synthetic Normal examples improve class coverage but are not a
replacement for naturally sampled neutral text. Annotation ambiguity between
urgency and scarcity can lower agreement. Results should therefore be reported
with per-class metrics, confusion matrices, calibration, and bootstrap
confidence intervals rather than accuracy alone.
