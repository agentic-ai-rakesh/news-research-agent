# RockyBot News Research Agent
# make setup      create .venv (Python 3.11+) and install requirements.txt
# make backend    FastAPI on http://127.0.0.1:$(BACKEND_PORT)  (docs at /docs)
# make frontend   Streamlit on http://localhost:$(FRONTEND_PORT)
# make run        both together (Ctrl+C stops both)
# make test       run the tests (no API key or network needed)

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(ROOT)/.venv/bin
PYTHON ?= python3.11
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 8501
export NEWS_RESEARCH_API_URL ?= http://127.0.0.1:$(BACKEND_PORT)

.PHONY: help setup setup-notebooks env backend frontend run test lint format clean

help:
	@sed -n 's/^# //p' $(MAKEFILE_LIST) | head -7

setup: env
	cd "$(ROOT)" && if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt; \
	else \
		$(PYTHON) -m venv .venv && .venv/bin/python -m pip install --upgrade pip && \
		.venv/bin/python -m pip install -r requirements.txt; \
	fi
	@echo "Setup done. Add your OPENAI_API_KEY to .env, then run: make run"

env:
	@cd "$(ROOT)" && if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example"; fi

setup-notebooks: setup
	cd "$(ROOT)" && .venv/bin/python -m pip install -r requirements-notebooks.txt 2>/dev/null || \
		uv pip install --python .venv/bin/python -r requirements-notebooks.txt

backend:
	cd "$(ROOT)" && "$(VENV)/python" -m uvicorn api.main:app --reload --host 127.0.0.1 --port $(BACKEND_PORT)

frontend:
	cd "$(ROOT)" && "$(VENV)/python" -m streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

run:
	cd "$(ROOT)" && \
	"$(VENV)/python" -m uvicorn api.main:app --host 127.0.0.1 --port $(BACKEND_PORT) & \
	BACKEND_PID=$$!; \
	trap 'kill $$BACKEND_PID 2>/dev/null; wait $$BACKEND_PID 2>/dev/null' EXIT INT TERM; \
	sleep 2; \
	cd "$(ROOT)" && "$(VENV)/python" -m streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

test:
	cd "$(ROOT)" && "$(VENV)/python" -m pytest -q

lint:
	cd "$(ROOT)" && "$(VENV)/ruff" check . && "$(VENV)/ruff" format --check .

format:
	cd "$(ROOT)" && "$(VENV)/ruff" check --fix . && "$(VENV)/ruff" format .

clean:
	cd "$(ROOT)" && rm -rf data logs .pytest_cache .ruff_cache && find . -name __pycache__ -prune -exec rm -rf {} +
