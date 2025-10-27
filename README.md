# 🏦 Loan Default Prediction - End-to-End MLOps Project

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/loan-default-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/loan-default-mlops/actions)
[![Docker Hub](https://img.shields.io/docker/pulls/zainulabidinshaikh/loan-default-app)](https://hub.docker.com/r/zainulabidinshaikh/loan-default-app)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

A production-ready machine learning system for predicting loan defaults with automated training, hyperparameter tuning, REST API, interactive UI, Docker containerization, and CI/CD pipeline.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Model Training](#-model-training)
- [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview

This project implements a complete MLOps pipeline for loan default prediction, from data preprocessing and model training to deployment and monitoring. The system uses advanced machine learning algorithms with automated hyperparameter optimization to predict the likelihood of loan defaults.

**Key Achievements:**

- ✅ Trained 5 ML algorithms with Optuna hyperparameter tuning
- ✅ Built FastAPI REST API for real-time predictions
- ✅ Created Streamlit web UI for non-technical users
- ✅ Containerized with Docker for portability
- ✅ Automated CI/CD pipeline with GitHub Actions
- ✅ Published Docker image to Docker Hub

---

## 🛠 Tech Stack

### **Machine Learning**

- **scikit-learn** - Classical ML algorithms (Logistic Regression, Random Forest, SVM)
- **CatBoost** - Gradient boosting for tabular data
- **LightGBM** - Fast gradient boosting framework
- **Optuna** - Automated hyperparameter optimization
- **Pandas & NumPy** - Data manipulation and numerical operations

### **Model Management**

- **MLflow** - Experiment tracking and model registry
- **Joblib** - Model serialization

### **Backend & API**

- **FastAPI** - High-performance REST API framework
- **Uvicorn** - ASGI server for production
- **Pydantic** - Data validation

### **Frontend**

- **Streamlit** - Interactive web UI

### **DevOps & Deployment**

- **Docker** - Containerization
- **GitHub Actions** - CI/CD automation
- **Docker Hub** - Container registry

### **Development Tools**

- **Python 3.10** - Programming language
- **Git** - Version control

---

## 📁 Project Structure

```
MLOps Project/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                 # GitHub Actions CI/CD pipeline
│
├── data/
│   └── loan_default_sample.csv       # Training dataset
│
├── exported_model/
│   └── best_loan_model/              # Trained MLflow model
│       ├── MLmodel
│       ├── conda.yaml
│       ├── model.pkl
│       └── python_env.yaml
│
├── mlruns/                           # MLflow experiment tracking
│
├── predictapi/
│   └── api.py                        # FastAPI application
│
├── ui/
│   └── streamlit_app.py              # Streamlit web interface
│
├── mytrain.py                        # Training script with Optuna
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── .dockerignore                     # Docker ignore rules
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---

## ✨ Features

### **1. Automated ML Pipeline**

- **5 ML Algorithms**: Logistic Regression, Random Forest, SVM, CatBoost, LightGBM
- **Hyperparameter Tuning**: Optuna-based automated optimization
- **Feature Engineering**: Automated feature creation and transformation
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### **2. REST API**

- **FastAPI Framework**: High-performance async API
- **Health Checks**: `/health` endpoint for monitoring
- **Swagger Documentation**: Auto-generated at `/docs`
- **Input Validation**: Pydantic-based request validation

### **3. Web Interface**

- **Interactive UI**: User-friendly Streamlit application
- **Real-time Predictions**: Instant results via API integration
- **Form Validation**: Client-side input validation

### **4. Containerization**

- **Docker Support**: Single-command deployment
- **Multi-service**: API + UI in one container
- **Portable**: Runs anywhere Docker is installed

### **5. CI/CD Pipeline**

- **Automated Testing**: Build and health checks
- **Docker Build**: Automatic image creation
- **Registry Push**: Auto-publish to Docker Hub
- **GitHub Actions**: Triggered on every push

---

## 🚀 Installation & Setup

### **Prerequisites**

- Python 3.10+
- Docker (optional, for containerized deployment)
- Git

### **Option 1: Local Development Setup**

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/loan-default-mlops.git
cd loan-default-mlops
```

2. **Create virtual environment:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### **Option 2: Docker Deployment (Recommended)**

**Pull and run the pre-built image:**

```bash
docker pull zainulabidinshaikh/loan-default-app:latest
docker run -p 9000:9000 -p 8501:8501 zainulabidinshaikh/loan-default-app:latest
```

**Or build locally:**

```bash
docker build -t loan-default-app .
docker run -p 9000:9000 -p 8501:8501 loan-default-app
```

---

## 📖 Usage Guide

### **1. Model Training**

#### **Train All Models (Recommended)**

```bash
python mytrain.py \
    --data-path ./data/loan_default_sample.csv \
    --target target_default \
    --model-type all \
    --mlflow-tracking-uri http://127.0.0.1:5000
```

#### **Train Specific Model**

```bash
# Logistic Regression
python mytrain.py --data-path ./data/loan_default_sample.csv --target target_default --model-type logistic

# Random Forest
python mytrain.py --data-path ./data/loan_default_sample.csv --target target_default --model-type rf

# SVM
python mytrain.py --data-path ./data/loan_default_sample.csv --target target_default --model-type svm

# CatBoost
python mytrain.py --data-path ./data/loan_default_sample.csv --target target_default --model-type catboost

# LightGBM
python mytrain.py --data-path ./data/loan_default_sample.csv --target target_default --model-type lightgbm
```

#### **Training Arguments**

| Argument                | Type | Default                 | Description                                                            |
| ----------------------- | ---- | ----------------------- | ---------------------------------------------------------------------- |
| `--data-path`           | str  | Required                | Path to training CSV file                                              |
| `--target`              | str  | Required                | Target column name                                                     |
| `--model-type`          | str  | `all`                   | Model to train: `all`, `logistic`, `rf`, `svm`, `catboost`, `lightgbm` |
| `--n-iter`              | int  | `10`                    | Number of Optuna optimization iterations                               |
| `--mlflow-tracking-uri` | str  | `http://127.0.0.1:5000` | MLflow server URL                                                      |

**Output:** Best model saved to `exported_model/best_loan_model/`

---

### **2. Start MLflow UI (Optional)**

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Access at: http://127.0.0.1:5000

---

### **3. Run FastAPI Server**

```bash
uvicorn predictapi.api:app --host 127.0.0.1 --port 9000
```

**Access:**

- API Root: http://127.0.0.1:9000
- Health Check: http://127.0.0.1:9000/health
- Swagger Docs: http://127.0.0.1:9000/docs

---

### **4. Run Streamlit UI**

```bash
streamlit run ui/streamlit_app.py
```

Access at: http://localhost:8501

---

### **5. Testing with Postman**

**Endpoint:** `POST http://127.0.0.1:9000/predict`

**Headers:**

```
Content-Type: application/json
```

**Body (JSON):**

```json
{
  "age": 32,
  "annual_income": 60000,
  "employment_length": 3,
  "home_ownership": "RENT",
  "purpose": "creditcard",
  "loan_amount": 15000,
  "term_months": 36,
  "interest_rate": 12.5,
  "dti": 20.3,
  "credit_score": 720,
  "delinquency_2yrs": 0,
  "num_open_acc": 6
}
```

**Response:**

```json
{
  "default_probability": 0.00514,
  "default_prediction": 0
}
```

---

## 📡 API Documentation

### **Endpoints**

#### **GET /**

Root endpoint - API status

#### **GET /health**

Health check endpoint

```json
{
  "model_loaded": true
}
```

#### **POST /predict**

Predict loan default probability

**Request Body:**
| Field | Type | Description |
|-------|------|-------------|
| age | int | Applicant age |
| annual_income | float | Annual income in dollars |
| employment_length | int | Years of employment |
| home_ownership | str | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| purpose | str | Loan purpose (e.g., `creditcard`, `debt_consolidation`) |
| loan_amount | float | Requested loan amount |
| term_months | int | Loan term (12, 24, 36, 60) |
| interest_rate | float | Interest rate percentage |
| dti | float | Debt-to-income ratio |
| credit_score | int | Credit score (300-850) |
| delinquency_2yrs | int | Number of delinquencies in past 2 years |
| num_open_acc | int | Number of open accounts |

**Response:**
| Field | Type | Description |
|-------|------|-------------|
| default_probability | float | Probability of default (0-1) |
| default_prediction | int | Binary prediction (0=safe, 1=default) |

---

## 🐳 Docker Deployment

### **Using Pre-built Image**

```bash
# Pull image
docker pull zainulabidinshaikh/loan-default-app:latest

# Run container
docker run -p 9000:9000 -p 8501:8501 zainulabidinshaikh/loan-default-app:latest
```

### **Building from Source**

```bash
# Build image
docker build -t loan-default-app .

# Run container
docker run -p 9000:9000 -p 8501:8501 loan-default-app

# Run in detached mode
docker run -d -p 9000:9000 -p 8501:8501 --name ml-app loan-default-app
```

### **Docker Commands Reference**

```bash
# List running containers
docker ps

# Stop container
docker stop ml-app

# Remove container
docker rm ml-app

# View logs
docker logs ml-app

# Execute command in container
docker exec -it ml-app bash
```

---

## 🔄 CI/CD Pipeline

### **Automated Workflow**

Every push to `main` branch triggers:

1. ✅ **Build & Test Job**

   - Install Python dependencies
   - Run linting checks
   - Build Docker image
   - Test container health

2. ✅ **Deploy Job** (on success)
   - Login to Docker Hub
   - Build production image
   - Push to Docker Hub with tags:
     - `latest`
     - `<commit-sha>`

### **GitHub Actions Setup**

**Required Secrets:**

- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub access token

**Workflow File:** `.github/workflows/ci-cd.yml`

---

## 🧠 Model Training

### **Algorithms Implemented**

1. **Logistic Regression**

   - Fast, interpretable baseline
   - L2 regularization

2. **Random Forest**

   - Ensemble of decision trees
   - Feature importance analysis

3. **Support Vector Machine (SVM)**

   - RBF kernel
   - Effective for non-linear boundaries

4. **CatBoost**

   - Gradient boosting
   - Handles categorical features natively

5. **LightGBM**
   - Fast gradient boosting
   - Memory efficient

### **Feature Engineering**

Automated feature creation including:

- `income_to_loan_ratio`
- `monthly_payment`
- `payment_to_income_ratio`
- `employment_risk`
- `young_borrower` / `senior_borrower`
- `high_credit_score` / `low_credit_score`
- `risk_score` (composite risk factor)
- Binned features (income, credit score, loan amount)
- Interaction features

### **Hyperparameter Optimization**

- **Framework:** Optuna
- **Iterations:** Configurable (default: 10)
- **Objective:** Maximize ROC-AUC score
- **Strategy:** Bayesian optimization

---

## 🔮 Future Enhancements

### **Planned Features**

- [ ] **AWS Deployment**

  - Deploy to AWS ECS/Fargate
  - Set up Application Load Balancer
  - Configure auto-scaling

- [ ] **Monitoring & Logging**

  - Prometheus metrics
  - Grafana dashboards
  - ELK stack for centralized logging

- [ ] **Model Improvements**

  - SHAP explainability
  - Model versioning with MLflow registry
  - A/B testing framework
  - Drift detection

- [ ] **Enhanced UI**

  - Batch predictions
  - Historical prediction logs
  - Model performance visualization

- [ ] **Security**
  - API authentication (JWT)
  - Rate limiting
  - Input sanitization

---

## 👨‍💻 Development

### **Local Development Workflow**

1. Make changes to code
2. Test locally:
   ```bash
   pytest tests/  # (if tests are added)
   ```
3. Build and test Docker image:
   ```bash
   docker build -t loan-default-app:dev .
   docker run -p 9000:9000 -p 8501:8501 loan-default-app:dev
   ```
4. Commit and push:
   ```bash
   git add .
   git commit -m "Your message"
   git push origin main
   ```

### **Environment Variables**

Create `.env` file (not committed to Git):

```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MODEL_PATH=exported_model/best_loan_model
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

**Zainulabidin Shaikh**

- GitHub: Zainulabidin Shaikh
- Email: zainulabidinshaikh12@gmail.com
- Docker Hub: [zainulabidinshaikh](https://hub.docker.com/u/zainulabidinshaikh)

---

## 🙏 Acknowledgments

- MLOps community for best practices
- FastAPI and Streamlit teams for excellent frameworks
- Optuna for hyperparameter optimization

---

**⭐ If you found this project helpful, please star it on GitHub!**
