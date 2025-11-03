.PHONY: install train evaluate api docker-build docker-up docker-down clean test

install:
	pip install -r requirements.txt

train:
	python src/pipeline.py

generate-data:
	python src/data/generator.py

extract-features:
	python src/features/extractor.py

train-model:
	python src/models/detector.py

evaluate:
	python src/evaluation/metrics.py

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

api-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --workers 4

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

test:
	pytest tests/ -v --cov=src

clean:
	rm -rf data/*.csv data/*.npy data/*.gpickle
	rm -rf models/*.pkl models/*.joblib
	rm -rf results/*.png results/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src/
	flake8 src/

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make train         - Run complete training pipeline"
	@echo "  make generate-data - Generate dataset only"
	@echo "  make train-model   - Train model only"
	@echo "  make evaluate      - Evaluate model"
	@echo "  make api           - Run API in development mode"
	@echo "  make api-prod      - Run API in production mode"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean generated files"
	@echo "  make format        - Format code"
