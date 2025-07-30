FROM python:3.10.11-slim

RUN apt-get update && apt-get install -y gcc && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN PIP_DEFAULT_TIMEOUT=100 PIPENV_TIMEOUT=100 pipenv install --system --deploy

COPY deployment ./api
COPY models ./models

EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "api"]

#docker build -t obesity-level-api .
#docker run -it -p 8080:8080 obesity-level-api

#gcloud builds submit --tag gcr.io/plucky-haven-463121-j1/obesity-level-api
#gcloud run deploy obesity-level-api --image gcr.io/plucky-haven-463121-j1/obesity-level-api --platform managed --region us-east1 --allow-unauthenticated --port 8080
