.PHONY: help setup api web test build clean fdb fdb-smoke fdb-replay

help:
	@echo "make setup  - install backend venv and frontend deps"
	@echo "make api    - run the agent server on :8000"
	@echo "make web    - run the web UI on :5174"
	@echo "make test   - run the backend test suite"
	@echo "make build  - production build of the web UI"
	@echo "make fdb    - full Full-Duplex-Bench v3 run (see README: Benchmark)"
	@echo "make fdb-smoke  - FDB-v3 run on 3 scenarios"
	@echo "make fdb-replay - offline text replay of FDB-v3 (no LiveKit)"

setup:
	cd server && python3.11 -m venv .venv && ./.venv/bin/pip install -q -r requirements-dev.txt
	cd web && npm install

api:
	cd server && ./.venv/bin/python -m uvicorn app.main:app --reload --port 8000

web:
	cd web && npm run dev

test:
	cd server && ./.venv/bin/python -m pytest tests -q

build:
	cd web && npm run build

fdb:
	./scripts/run_fdb_v3.sh

fdb-smoke:
	./scripts/run_fdb_v3.sh --subset 3

fdb-replay:
	cd server && ./.venv-fdb/bin/python -m app.fdb.replay --one-per-scenario

clean:
	rm -rf server/.venv server/.pytest_cache web/node_modules web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
