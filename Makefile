# Colab: !git clone <repo> && cd project && pip install -r requirements.txt && make neural
.PHONY: setup data baselines neural llm ensemble report app frontend frontend-build test lint smoke format format-check

setup:
	python -m pip install -r requirements.txt

data:
	python -m data.clean
	python -m data.split

baselines:
	python -m models.classical

neural:
	python -m models.cnn
	python -m models.bilstm
	python -m models.small_transformer
	python -m models.finetune --model all

llm:
	python -m models.llm --variant all

ensemble:
	python -m ensemble.ensemble

report:
	python -m eval.benchmark

app:
	uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

frontend:
	cd app/frontend && npm run dev

frontend-build:
	cd app/frontend && npm run build && npm run lint

test:
	python -m pytest

lint:
	python -m ruff check .

format:
	python -m black .

format-check:
	python -m black --check .

smoke:
	python scripts/smoke.py
