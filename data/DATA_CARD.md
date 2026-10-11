# DarkPatternGuard data card

The raw source is the `dataset/dataset.tsv` release from [yamanalab/ec-darkpattern](https://github.com/yamanalab/ec-darkpattern), pinned to commit `85ea9df31928d5ad6ea0b106316b441c201adbbe`. Its README describes the dark-pattern examples as originating from Mathur et al. (2019) and its Normal examples as non-dark e-commerce segments collected from the same sites. The repository includes an Apache-2.0 license.

- Total rows: 2,029
- Seed: 1729
- Unique source IDs: 956
- Synthetic rows: 0
- Split by source page to reduce page-level leakage.
- Labels: Normal, Scarcity, Social Proof, Urgency

## Grouped split sizes

| Split | Rows |
| --- | ---: |
| Train | 1,610 |
| Val | 200 |
| Test | 219 |

## Class distribution

| Label | Count | Weight |
| --- | ---: | ---: |
| Normal | 1163 | 0.436156 |
| Scarcity | 349 | 1.453438 |
| Social Proof | 312 | 1.625801 |
| Urgency | 205 | 2.474390 |

Only Urgency, Scarcity, Social Proof, and `Not Dark Pattern` records are retained; `Not Dark Pattern` maps to Normal. Other source categories are excluded. Original text and source-page/category metadata remain available in the raw JSON; the cleaner records the source ID and category in the cleaned CSV. No synthetic examples were added when genuine Normal examples met the balancing threshold.

## Citation

Yada et al., *Dark patterns in e-commerce: a dataset and its baseline evaluations* (2022). The dataset is derived in part from Mathur et al., *Dark Patterns at Scale* (2019).
