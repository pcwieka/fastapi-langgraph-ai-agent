.PHONY: build up down logs test lint format clean eval

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f

test:
	python -m pytest tests/ -v

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

eval:
	python evaluation/skill_router.eval.py

clean:
	docker compose down -v
