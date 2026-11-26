# ♻️ EcoSort AI — Intelligent Waste Classification and Sorting System

**A machine learning-powered waste classification system designed to support smarter waste separation and disposal.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![Deployment](https://img.shields.io/badge/Deployment-Oracle%20Cloud-red)](https://www.oracle.com/cloud/)

---

## 1. Project Overview

EcoSort AI is an intelligent waste classification system that uses computer vision and deep learning to identify waste materials from images and provide corresponding disposal recommendations.

The project addresses a practical waste-management challenge: waste materials are frequently placed in incompatible disposal bins, contributing to contamination and reducing the effectiveness of recycling and waste recovery processes.

EcoSort AI explores how machine learning can support better waste separation by automatically recognizing waste materials and communicating the appropriate disposal stream.

The system combines a trained MobileNetV2 image classification model, a FastAPI prediction service, a Streamlit dashboard, and a Docker-based deployment architecture.

### Live Application

| Component | URL |
|---|---|
| **EcoSort AI Dashboard** | http://148.116.88.130:8501 |
| **API Health Check** | http://148.116.88.130:8000/health |
| **Interactive API Documentation** | http://148.116.88.130:8000/docs |
| **OpenAPI Schema** | http://148.116.88.130:8000/openapi.json |

**Deployment status:** Running on Oracle Cloud Infrastructure.

The API health endpoint reports the active model version, model loading status, and service uptime.

### Demo Video

<!-- Add your YouTube demonstration URL here -->

Demo vid[]([url](https://drive.google.com/file/d/1w7qPB3E9ZJqNZuqJuv5TtzoojvoAxkFr/view?usp=sharing))

---

## 2. Problem Statement

Waste management systems depend on proper waste separation. However, recyclable, organic, and non-recyclable materials can end up in incompatible disposal bins.

This creates challenges such as:

- Contamination of recyclable materials.
- Inefficient waste collection and sorting.
- Reduced recovery of reusable materials.
- Increased pressure on waste-management systems.

EcoSort AI investigates the use of image classification to identify waste materials and provide actionable disposal guidance.

### Proposed Solution

The system follows this process:

1. A user uploads an image of a waste item.
2. The image is processed and passed to the trained classification model.
3. The model predicts the most likely waste category.
4. The system calculates prediction confidence.
5. A configurable disposal rule maps the predicted class to a disposal stream.
6. The dashboard displays the classification and recommendation.

The system also supports collecting additional labelled images for controlled model retraining.

---

## 3. Recognized Waste Categories

EcoSort AI currently recognizes nine waste categories:

| No. | Waste Category |
|---|---|
| 1 | Cardboard |
| 2 | Food Organics |
| 3 | Glass |
| 4 | Metal |
| 5 | Miscellaneous Trash |
| 6 | Paper |
| 7 | Plastic |
| 8 | Textile Trash |
| 9 | Vegetation |

The disposal mapping is configurable through `disposal_rules.json`. Actual disposal practices should be validated against local waste-management requirements.

---

## 4. System Architecture

EcoSort AI uses a containerized architecture with three main application services.

```text
                    USER
                      |
                      v
              STREAMLIT DASHBOARD
                  Port 8501
                      |
                      v
               NGINX REVERSE PROXY
                  Port 8000
                      |
                      v
                 FASTAPI SERVICE
                      |
                      v
              IMAGE PREPROCESSING
                      |
                      v
              MOBILENETV2 MODEL
                      |
                      v
              CLASSIFICATION RESULT
                      |
                      v
             DISPOSAL RECOMMENDATION
```

### Main Components

| Component | Responsibility |
|---|---|
| Streamlit | User interface, monitoring, visualizations, uploads and retraining controls |
| FastAPI | Image prediction and health-check endpoints |
| Nginx | Reverse proxy for API traffic |
| TensorFlow / Keras | Deep learning inference and model training |
| MobileNetV2 | Image feature extraction and classification |
| Docker Compose | Container orchestration |
| Locust | Load and performance testing |
| Oracle Cloud | Public cloud deployment |

### Docker Services

The `docker-compose.yml` configuration defines:

- `api` — FastAPI application.
- `nginx` — reverse proxy.
- `dashboard` — Streamlit application.
- `locust` — optional load-testing service using the `loadtest` profile.

---

## 5. Machine Learning Model

### Architecture

The classification model uses MobileNetV2 with ImageNet pretrained weights.

The architecture consists of:

1. MobileNetV2 feature extraction.
2. Global Average Pooling.
3. Dropout regularization.
4. Dense softmax classification layer.

**Model input:** 224 × 224 × 3 RGB images.

**Input pixel range:** 0–255, with scaling performed inside the model.

**Model artifact:** `models/ecosort_mobilenetv2_v2.keras`

**Active model version:** v2.

### Dataset Splits

The recorded dataset split contains:

| Dataset | Images |
|---|---:|
| Training | 3,325 |
| Validation | 716 |
| Testing | 711 |
| **Total** | **4,752** |

Random seed: `42`.

### Model Evaluation

The original independently evaluated model produced the following held-out test results:

| Metric | Result |
|---|---:|
| Test Accuracy | 81.86% |
| Macro F1 Score | 82.25% |

These metrics represent the original independently evaluated model. They should not be interpreted as a new test evaluation of the subsequently promoted v2 model.

### Model Retraining

The system supports controlled retraining using additional labelled images.

The recorded v2 retraining experiment used 45 uploaded images.

| Metric | Baseline | Candidate |
|---|---:|---:|
| Validation Loss | 0.5195 | 0.5058 |
| Validation Accuracy | 81.84% | 82.96% |

The candidate completed four training epochs and passed the promotion criteria.

The promotion gate requires:

- Candidate validation loss to be lower than baseline validation loss.
- Candidate validation accuracy to remain within the configured tolerance of baseline accuracy.

The test set is not used during candidate promotion.

**Important:** The 82.96% candidate result is validation accuracy, not test accuracy.

---

## 6. Application Features

### Dashboard and Monitoring

The Streamlit dashboard provides:

- API availability monitoring.
- Model loading status.
- API uptime.
- Active model version.
- Confidence threshold.
- Evaluation metrics.
- Training data visualizations.

### Visualizations

The dashboard includes:

1. Training images per class.
2. Training and validation learning curves.
3. Held-out test confusion matrix.

These visualizations help interpret dataset distribution, model learning behavior, and classification errors.

### Single Image Classification

Users can upload an individual waste image and obtain:

- Predicted material category.
- Prediction confidence.
- Top predictions.
- Configured disposal stream.
- Manual-review indication when appropriate.

### Bulk Labelled Image Upload

Users can upload a ZIP archive containing labelled image folders.

The application validates image files and stages accepted images for future retraining.

Staged images do not immediately modify the active model.

### Controlled Retraining

The dashboard includes a retraining control that:

1. Uses the active model as the starting point.
2. Combines the original training data with staged labelled images.
3. Trains a candidate model.
4. Evaluates the candidate using validation data.
5. Applies the promotion criteria.
6. Promotes the candidate only when it passes validation.

If the candidate fails the promotion criteria, the active model is retained.

---

## 7. API Documentation

The FastAPI service exposes the following application endpoints.

Base URL:

```text
http://148.116.88.130:8000
```

### GET `/health`

Checks service health, model loading status, model version and uptime.

Example:

```bash
curl http://148.116.88.130:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v2",
  "uptime_seconds": 457.71
}
```

### POST `/predict`

Classifies a single uploaded image.

**Request:** Multipart form data.

| Field | Type | Description |
|---|---|---|
| `file` | Image file | Waste image to classify |

Example:

```bash
curl -X POST \
  -F "file=@plastic_bottle.jpg" \
  http://148.116.88.130:8000/predict
```

The response includes the uploaded filename and prediction information, including:

- `predicted_class`
- `confidence`
- `top_k`
- `uncertain`
- `disposal_stream`
- `manual_review`
- `model_version`

### POST `/predict/batch`

Processes multiple uploaded images.

**Request:** Multipart form data.

| Field | Type | Description |
|---|---|---|
| `files` | Multiple image files | Images to classify |

Example:

```bash
curl -X POST \
  -F "files=@plastic.jpg" \
  -F "files=@cardboard.jpg" \
  http://148.116.88.130:8000/predict/batch
```

The response contains a count and individual results for the submitted files.

### Interactive Documentation

FastAPI automatically generates interactive API documentation:

http://148.116.88.130:8000/docs

---

## 8. Repository Structure

```text
EcoSort-AI/
│
├── notebooks/
│   ├── 01_download_and_inspect_FIXED (1).ipynb
│   ├── 02_preprocessing_and_splits_FIXED (1).ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   ├── 04_application_integration_and_prediction_api.ipynb
│   └── 05_dashboard_and_retraining.ipynb
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── dashboard.py
│   ├── predictor.py
│   └── retrain.py
│
├── models/
│   ├── ecosort_mobilenetv2_v2.keras
│   └── model_metadata.json
│
├── splits/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── train_with_groups.csv
│   ├── val_with_groups.csv
│   ├── test_with_groups.csv
│   ├── class_names.json
│   └── class_weights.json
│
├── reports/
│   ├── metrics.json
│   ├── last_retraining.json
│   └── figures/
│
├── disposal_rules.json
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── locustfile.py
├── requirements-api.txt
├── requirements-dashboard.txt
├── manifest.csv
├── .dockerignore
└── .gitignore
```

Large raw datasets and selected intermediate model artifacts are excluded from version control.

---

## 9. Local Installation and Setup

### Requirements

- Python 3.11
- Docker Engine
- Docker Compose V2
- Git

### Clone the Repository

```bash
git clone https://github.com/Hirwavsopps/EcoSort-AI.git

cd EcoSort-AI
```

### Build and Start the Application

```bash
docker compose up -d --build
```

### Check Running Services

```bash
docker compose ps
```

### Access the Application

| Component | Local URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API Health | http://localhost:8000/health |
| API Documentation | http://localhost:8000/docs |

### Stop the Application

```bash
docker compose down
```

---

## 10. Oracle Cloud Deployment

EcoSort AI is deployed on Oracle Cloud Infrastructure using an Ubuntu 22.04 virtual machine.

### Deployment Configuration

| Setting | Configuration |
|---|---|
| Cloud provider | Oracle Cloud Infrastructure |
| Operating system | Ubuntu 22.04 |
| Compute shape | VM.Standard.E4.Flex |
| OCPU | 1 |
| Memory | 16 GB |
| Container runtime | Docker |
| Orchestration | Docker Compose |
| Reverse proxy | Nginx |

### Deployment Steps

Install Docker and supporting tools:

```bash
sudo apt update

sudo apt install -y docker.io docker-compose-v2 git curl
```

Enable Docker:

```bash
sudo systemctl enable --now docker

sudo usermod -aG docker ubuntu
```

Reconnect to the server, then clone the repository:

```bash
git clone https://github.com/Hirwavsopps/EcoSort-AI.git

cd EcoSort-AI
```

Build and start the application:

```bash
docker compose up -d --build
```

Check service status:

```bash
docker compose ps
```

### Network Configuration

The Oracle Cloud subnet security list must allow inbound TCP traffic on:

| Port | Purpose |
|---|---|
| 22 | SSH administration |
| 8000 | API and Nginx |
| 8501 | Streamlit dashboard |

For production use, configure HTTPS, restrict administrative access, and use appropriate network security rules.

---

## 11. Performance Testing with Locust

Locust is included to simulate multiple users making prediction requests against the API.

The load-testing configuration is defined in `locustfile.py`.

### Locust Architecture

```text
             LOCUST USERS
                   |
                   v
              NGINX PROXY
                   |
                   v
              FASTAPI API
                   |
                   v
             MODEL INFERENCE
```

### Starting Locust

The optional Locust service can be started using:

```bash
docker compose --profile loadtest up -d locust
```

Open the Locust interface:

```text
http://localhost:8089
```

The Locust configuration sends image prediction requests to `/predict`.

**Dataset requirement:** The Locust test script reads image paths from `splits/test.csv` and expects the corresponding RealWaste images under `data_raw/realwaste-main/RealWaste`. The dataset must be available in the environment running Locust.

### Recorded Load-Test Results

The following results were recorded during earlier load tests against the Nginx API endpoint.

| Virtual Users | Requests | Average Response Time | P95 Latency | Failures | Requests/sec |
|---:|---:|---:|---:|---:|---:|
| 5 | 1,070 | 297.47 ms | 480 ms | 0 | 2.2 |
| 10 | 3,066 | 467.02 ms | 820 ms | 0 | 4.0 |
| 20 | 614 | 1,779.82 ms | 2,400 ms | 0 | 5.3 |
| 50 | 99 | 4,205.68 ms | 8,400 ms | 0 | 5.0 |

### Interpretation

The recorded tests show that:

- The API handled the tested requests without reported failures.
- Average latency increased as concurrent virtual users increased.
- Throughput approached approximately 5 requests per second in the higher-concurrency tests.
- Response times increased substantially under heavier load.

These results suggest that model inference becomes a performance bottleneck as concurrency increases.

**Testing distinction:** The table reports different virtual-user loads. It does not represent a controlled comparison of one, two, and three API container replicas.

---

## 12. Limitations

The current implementation has several limitations:

- Image classification depends on the quality and diversity of the training dataset.
- Visually similar materials can be confused by the model.
- Prediction confidence does not guarantee that a classification is correct.
- Disposal recommendations depend on locally configured rules.
- The model has not been independently re-evaluated on the held-out test set after the v2 promotion.
- The public deployment currently uses HTTP rather than HTTPS.
- Bulk uploads require correct image labels to support meaningful retraining.
- The system is a prototype and has not yet been validated as a physical automated sorting machine.

The project demonstrates the software and machine learning pipeline. Physical sorting hardware and mechanical bin actuation are outside the current deployed implementation.

---

## 13. Future Improvements

Potential future development includes:

- Adding more diverse waste images.
- Improving classification performance for visually similar categories.
- Evaluating subsequent model versions on an independent test set.
- Introducing HTTPS and stronger API security.
- Adding persistent, monitored retraining workflows.
- Supporting additional waste categories.
- Integrating the classifier with physical smart-bin hardware.
- Exploring automated mechanical sorting and compartment selection.
- Conducting further load tests with different API replica counts.

---

## 14. Academic Context

**Institution:** African Leadership University

**Programme:** Bachelor of Software Engineering (BSE)

**Course:** Machine Learning Pipeline

**Project:** EcoSort AI

This project demonstrates an end-to-end machine learning workflow, including data acquisition, preprocessing, model training, evaluation, application integration, API development, dashboard development, containerization, cloud deployment and performance testing.

---

## 15. License and Acknowledgements

This project was developed for academic purposes as part of the Machine Learning Pipeline summative assignment.

Dataset sources and relevant third-party resources should be acknowledged according to their respective licenses and usage requirements.

---

**EcoSort AI — Making waste classification smarter, one image at a time.** ♻️
