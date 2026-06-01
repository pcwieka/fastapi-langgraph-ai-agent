.PHONY: build up down logs test clean

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

clean:
	docker compose down -v
