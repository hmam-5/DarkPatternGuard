# Humam Tahseen Abdullah Al Mohammed — 240103130

## Contribution statement

I contributed across the complete DarkPatternGuard stack:

- designed the reproducible repository structure, Python 3.11 tooling,
  configuration, Make targets, formatting, linting, and tests;
- implemented the data cleaning and stratified split pipeline, including HTML
  and URL removal, Unicode normalization, exact deduplication, Normal-class
  balancing, manifests, class weights, and the data card;
- implemented shared evaluation with macro-F1, micro-F1, per-class metrics,
  calibration error, confusion matrices, bootstrap confidence intervals,
  paired bootstrap testing, standardized prediction artifacts, benchmark
  tables, plots, and error analysis;
- implemented classical baselines, Kim CNN, BiLSTM, custom-BPE Transformer,
  Hugging Face fine-tuning, cached GPT-4o-mini zero-shot/five-shot evaluation,
  validation-only calibration, voting, weighting, stacking, and ablations;
- built the FastAPI inference backend with lazy model registry loading,
  explainability fields, request metrics, structured metadata logging,
  privacy-aware SHA-256 persistence, history, feedback, statistics, SQLAlchemy
  models, and Alembic migrations;
- built the React, TypeScript, Vite, Tailwind, Lucide, and Playwright frontend
  with analyze, batch comparison, history, benchmark, dark mode, typed API
  calls, error states, and CSV export;
- added Docker images, Compose healthchecks, GitHub Actions, pre-commit
  hooks, dependency audits, model prefetching, documentation, and release
  hygiene.

## Engineering practices

The implementation keeps validation separate from the held-out test set,
centralizes random seeds, avoids committing model weights or secrets, and
verifies changes with Black, Ruff, pytest, TypeScript compilation, frontend
builds, and repository audits.
