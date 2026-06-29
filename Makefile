.PHONY: install test ask eval api lint
install:
	pip install -e ".[dev]"
test:
	PYTHONPATH=src pytest -q
ask:
	PYTHONPATH=src python -m production_agentic_rag.cli ask "what is the return window?"
eval:
	PYTHONPATH=src python -m production_agentic_rag.cli eval
api:
	PYTHONPATH=src uvicorn api.main:app --reload
