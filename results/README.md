# Results

Store reproducible benchmark summaries and figures here. Do not store model
weights or caches.

The root leaderboard and plots are generated from the grouped real-data test
split. The leaderboard reports mean test Macro-F1 across configured seeds;
`errors.md` includes per-model error summaries and examples where DeBERTa
corrects BERT errors. Per-model and ensemble directories contain both
`val_preds.csv` and test `preds.csv` for each seed, along with test metrics and
confusion matrices. Use validation predictions for calibration/model selection
and test predictions for final evaluation.

Smoke-run summary reports and plots are archived under `smoke_run/` and are not
real-dataset evaluation results.
