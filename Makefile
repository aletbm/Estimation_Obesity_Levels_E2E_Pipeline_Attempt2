.PHONY: install test lint run-monitoring run-api run-training build-image push-image terraform-init terraform-apply terraform-destroy

IMAGE_NAME=gcr.io/plucky-haven-463121-j1/obesity-level-api
TAG=latest

install:
	pip install pipenv
	pipenv install --deploy --ignore-pipfile

shell:
	pipenv shell

test:
	pytest tests/

lint:
	pre-commit run --all-files

run-monitoring:
	python monitoring/monitor.py $(ALIAS)

build-image:
	docker build -t obesity-level-api .

gcloud-build:
	gcloud builds submit --tag $(IMAGE_NAME):$(TAG) .

run-api:
	docker build -t obesity-level-api .
	docker run --rm -p 8080:8080 obesity-level-api

run-prefect:
	prefect server start

run-mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models

run-training:
	python pipelines/training_flow.py $(ALIAS)

list-models:
	python scripts/list_registered_models.py

run-inference:
	python pipelines/batch_inference.py  data/test.parquet

terraform-deploy:
	terraform -chdir=infra init
	terraform -chdir=infra plan
	terraform -chdir=infra apply -auto-approve

terraform-destroy:
	terraform -chdir=infra destroy -auto-approve

test-remote:
	python deployment/test_serve.py


GCP_PROJECT_ID = plucky-haven-463121-j1
SERVICE_NAME = obesity-level-api
REGION = us-east1
IMAGE = gcr.io/$(GCP_PROJECT_ID)/$(SERVICE_NAME)
PORT = 8080

deploy-gcp:
	gcloud builds submit --tag $(IMAGE)
	gcloud run deploy $(SERVICE_NAME) \
		--image $(IMAGE) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--port $(PORT)