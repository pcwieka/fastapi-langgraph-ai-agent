.PHONY: build up down logs test lint format clean

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose exec ecommerce-agent python -m pytest tests/ -v

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

clean:
	docker compose down -v
