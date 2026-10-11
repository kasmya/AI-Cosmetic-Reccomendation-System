.PHONY: install train run test eval lint docker
install:
	pip install -r requirements.txt -r requirements-dev.txt
train:
	python -m cosmetic_rec.train
run:
	uvicorn cosmetic_rec.api:app --reload
test:
	pytest -q
eval:
	python -m cosmetic_rec.evaluate
lint:
	ruff check .
docker:
	docker build -t cosmetic-rec . && docker run -p 8000:8000 cosmetic-rec
