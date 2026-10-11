.PHONY: install train retrain registry run test eval lint docker up loadtest
install:
	pip install -r requirements.txt -r requirements-dev.txt
train:                      ## train a challenger; the registry gate decides whether it becomes champion
	python -m cosmetic_rec.train
retrain:                    ## the same, also learning from stored thumbs up/down
	python -m cosmetic_rec.train --with-feedback
registry:
	python -m cosmetic_rec.registry list
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
up:                         ## API + PostgreSQL + Prometheus + Grafana
	docker compose up --build
loadtest:                   ## server must already be running on :8000
	pip install -r requirements-loadtest.txt
	locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000
