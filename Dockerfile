
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ECOSORT_PROJECT_DIR=/app
ENV MPLBACKEND=Agg

COPY requirements-api.txt requirements-api.txt
COPY requirements-dashboard.txt requirements-dashboard.txt

RUN pip install --no-cache-dir \
    -r requirements-api.txt \
    -r requirements-dashboard.txt

COPY . .

EXPOSE 8000 8501
