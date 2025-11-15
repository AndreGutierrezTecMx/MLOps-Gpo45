# 🧠 MLOps Project – Predicting Article Popularity on Social Networks

## 📋 Project Overview
This project aims to **predict the number of social media shares** of online articles published by **Mashable** over a two-year period.  
The dataset **summarizes a heterogeneous set of features** about each article – including textual, temporal, and contextual metrics – to estimate its potential **popularity**.

> **Goal:** Build a complete MLOps pipeline to manage data, model development, and deployment efficiently.

---

## 🗂️ Dataset Description
- **Source:** Mashable online articles dataset  
- **Period:** Two years of publication data  
- **Target Variable:** Number of shares per article  
- **Features:**  
  - Article metadata (title, keywords, category, etc.)  
  - Temporal and channel-related features  
  - Textual content and sentiment analysis metrics  

> 🧩 The dataset represents a **heterogeneous feature space**, ideal for regression tasks predicting continuous popularity metrics.

---

## ⚙️ Team Organization

| Role | Name | Responsibilities |
|------|------|------------------|
| 🧑‍🔬 **Data Scientist** | **Pedro** | - Requirements Analysis<br>- Prediction of article shares |
| 🧠 **Data Scientist** | **Alex** | - Data Manipulation<br>- Feature Engineering & Preparation |
| 🧑‍💻 **Data Engineer / Data Scientist** | **Héctor** | - Data Exploration<br>- Data Preprocessing |
| 🧑‍🚀 **DevOps** | **Andre** | - Data Versioning<br>- Pipeline Automation |
| 🤖 **ML Engineer** | **Carlos** | - Model Construction<br>- Hyperparameter Tuning<br>- Model Evaluation |

---

## 📁 Project Structure

The project follows a **hierarchical and modular structure** designed for MLOps best practices:

```
project-root/
│
├── configs/                 # Configuration and dependency files
│
├── data/                    # Data storage directory
│   ├── raw/                 # Raw datasets (unmodified)
│   ├── interim/             # Intermediate datasets (partially cleaned)
│   └── processed/           # Final processed datasets ready for modeling
│
├── docs/                    # Documentation, reports, and project information
│
├── models/                  # Trained models and serialized versions
│
├── notebooks/               # Experimental and exploratory notebooks
│
├── src/                     # Main source code directory
│   ├── constants/           # Global constants and project paths
│   ├── data/                # Data processing scripts
│   ├── modeling/            # Model training, validation, and inference code
│   ├── utils/               # Helper functions and utility classes
│   └── versioning/          # DVC and MLflow version control configurations
│
├── tests/                   # 🆕 Automated testing suite
│   ├── conftest.py          # Shared fixtures for all tests
│   ├── test_data_cleaning.py
│   ├── test_data_preprocessing.py
│   ├── test_data_reader.py
│   ├── test_data_explorer.py
│   ├── test_data_analysis.py
│   └── test_integration_pipeline.py
│
├── pytest.ini               # 🆕 Pytest configuration
├── run_tests.py             # 🆕 Test execution utility script
├── TESTING_README.md        # 🆕 Complete testing documentation
└── README.md                # Main project documentation
```

> 🧭 **Note:** This structure supports reproducibility and scalability with tools like **DVC**, **MLflow**, and **Docker**.

---

## 🧪 Testing & Quality Assurance

This project implements a **comprehensive automated testing suite** to ensure code quality and pipeline reliability.

### 📊 Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| **Unit Tests** | 128 | ✅ All Passing |
| **Integration Tests** | 35 | ✅ All Passing |
| **Total** | **163** | ✅ **100% Pass Rate** |

### 🚀 Running Tests

#### Quick Start
```bash
# Install testing dependencies
pip3 install pytest pytest-cov

# Run all tests
python3 -m pytest -v

# Run with coverage report
python3 -m pytest --cov=src --cov-report=html
```

#### Run Specific Test Suites
```bash
# Unit tests only
python3 -m pytest tests/test_data_cleaning.py -v
python3 -m pytest tests/test_data_preprocessing.py -v

# Integration tests only
python3 -m pytest tests/test_integration_pipeline.py -v

# Quick smoke tests
python3 -m pytest -k "initialization" -v
```

#### Using the Test Runner Script
```bash
# All tests with detailed output
python run_tests.py

# Quick tests (unit tests only)
python run_tests.py --quick

# With coverage report
python run_tests.py --coverage

# Integration tests only
python run_tests.py --integration
```

### 📋 Test Components

**Unit Tests:**
- ✅ Data Reading & Loading (`test_data_reader.py`)
- ✅ Data Exploration (`test_data_explorer.py`)
- ✅ Data Cleaning (`test_data_cleaning.py`)
- ✅ Data Preprocessing (`test_data_preprocessing.py`)
- ✅ Data Analysis (`test_data_analysis.py`)

**Integration Tests:**
- ✅ End-to-End Pipeline (`test_integration_pipeline.py`)
- ✅ Reader → Cleaning → Preprocessing → Model
- ✅ Reproducibility Validation
- ✅ Error Handling & Edge Cases

### 📚 Documentation

For complete testing documentation, see:
- **[TESTING_README.md](TESTING_README.md)** - Comprehensive testing guide
- **[GUIA_ACTIVACION_TESTS.md](GUIA_ACTIVACION_TESTS.md)** - Step-by-step activation guide

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip or conda package manager

### Install Dependencies

```bash
# Install core dependencies
pip3 install -r requirements.txt

# Or install from dependencies.json
pip3 install dvc pandas scikit-learn dvc[gdrive] cryptography category_encoders mlflow xgboost==2.0.1

# Install testing dependencies
pip3 install pytest pytest-cov pytest-xdist
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MLOps-Gpo45
```

### 2. Set Up Environment
```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
```

### 3. Run Data Pipeline
```bash
# Execute data processing scripts
python src/data/data_reader.py
python src/data/data_cleaning.py
python src/data/data_preprocessing.py
```

### 4. Run Tests
```bash
# Verify everything works
python3 -m pytest -v
```

### 5. Train Models
```bash
# Execute model training
python src/modeling/modeling_pipeline.py
```

---

## � Docker Deployment

### Quick Start with Docker

The project includes a complete Docker setup that automatically:
1. ✅ Installs all dependencies
2. ✅ Trains the ML model
3. ✅ Starts the FastAPI server for predictions

#### Prerequisites
- Docker installed on your system ([Download Docker](https://www.docker.com/products/docker-desktop))

#### Build the Docker Image

```bash
# Navigate to project root
cd MLOps-Gpo45

# Build the image
docker build -t ml-service:latest .
```

#### Run the Container

```bash
# Run the container with FastAPI on port 8000
docker run -p 8000:8000 ml-service:latest
```

**Expected Output:**
```
Entrenando modelo...
[Training logs will appear here...]
Iniciando API en puerto 8000...
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

#### Access the API

Once the container is running:

- **API Base URL:** `http://localhost:8000`
- **Interactive API Docs (Swagger):** `http://localhost:8000/docs`
- **Alternative Docs (ReDoc):** `http://localhost:8000/redoc`

#### API Endpoints Examples

**1. GET `/` - Root Endpoint**

Get basic API information:

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "MLOps-Gpo45 Model API",
  "version": "1.0.0"
}
```

**2. GET `/model-info` - Model Information**

Get details about the loaded model:

```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "run_id": "abc123def456...",
  "run_name": "HistGradientBoosting (Poisson)",
  "metric": "R2",
  "metric_value": 0.85
}
```

**3. POST `/predict` - Make Predictions**

Send article data to get predictions:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
  "data": [
    {
      "n_tokens_title": 10,
      "n_tokens_content": 500,
      "n_unique_tokens": 0.5,
      "n_non_stop_words": 0.8,
      "n_non_stop_unique_tokens": 0.6,
      "num_hrefs": 5,
      "num_self_hrefs": 2,
      "num_imgs": 3,
      "num_videos": 0,
      "average_token_length": 4.5,
      "num_keywords": 5,
      "data_channel_is_lifestyle": 0,
      "data_channel_is_entertainment": 1,
      "data_channel_is_bus": 0,
      "data_channel_is_socmed": 0,
      "data_channel_is_tech": 0,
      "data_channel_is_world": 0,
      "kw_min_min": 0,
      "kw_max_min": 0,
      "kw_avg_min": 0,
      "kw_min_max": 0,
      "kw_max_max": 0,
      "kw_avg_max": 0,
      "kw_min_avg": 0,
      "kw_max_avg": 0,
      "kw_avg_avg": 0,
      "self_reference_min_shares": 0,
      "self_reference_max_shares": 0,
      "self_reference_avg_sharess": 0,
      "weekday_is_monday": 0,
      "weekday_is_tuesday": 0,
      "weekday_is_wednesday": 0,
      "weekday_is_thursday": 0,
      "weekday_is_friday": 0,
      "weekday_is_saturday": 0,
      "weekday_is_sunday": 0,
      "is_weekend": 0,
      "LDA_00": 0.2,
      "LDA_01": 0.3,
      "LDA_02": 0.1,
      "LDA_03": 0.2,
      "LDA_04": 0.2,
      "global_subjectivity": 0.5,
      "global_sentiment_polarity": 0.1,
      "global_rate_positive_words": 0.3,
      "global_rate_negative_words": 0.1,
      "rate_positive_words": 0.75,
      "rate_negative_words": 0.25,
      "avg_positive_polarity": 0.4,
      "min_positive_polarity": 0.2,
      "max_positive_polarity": 0.6,
      "avg_negative_polarity": -0.3,
      "min_negative_polarity": -0.5,
      "max_negative_polarity": -0.1,
      "title_subjectivity": 0.4,
      "title_sentiment_polarity": 0.2,
      "abs_title_subjectivity": 0.4,
      "abs_title_sentiment_polarity": 0.2,
      "article_year": 2013,
      "timedelta":231,
      "article_month":11,
      "article_day":1
    }
  ]
}'
```

**Response:**
```json
{
  "predictions": [3456.78]
}
```

#### Advanced Docker Usage

**Run container in background (detached mode):**
```bash
docker run -d -p 8000:8000 --name ml-service ml-service:latest
```

**View container logs:**
```bash
docker logs ml-service
```

**Stop the container:**
```bash
docker stop ml-service
```

**Remove the container:**
```bash
docker rm ml-service
```

**View running containers:**
```bash
docker ps
```

---

## �🔧 MLOps Tools & Technologies

| Tool | Purpose | Status |
|------|---------|--------|
| **DVC** | Data Version Control | ✅ Implemented |
| **MLflow** | Experiment Tracking | ✅ Implemented |
| **pytest** | Automated Testing | ✅ Implemented |
| **scikit-learn** | ML Modeling | ✅ Implemented |
| **pandas** | Data Processing | ✅ Implemented |
| **XGBoost** | Gradient Boosting | ✅ Implemented |
| **Docker** | Containerization | ✅ Implemented |
| **FastAPI** | Model Serving | ✅ Implemented |

---

## 📈 Pipeline Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Data Reader │ --> │ Data Cleaner │ --> │ Preprocessor    │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                   │
                                                   v
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Serving   │ <-- │    Model     │ <-- │ Model Training  │
└─────────────┘     └──────────────┘     └─────────────────┘
```

1. **Data Ingestion** - Load raw data from DVC repository
2. **Data Cleaning** - Handle missing values, outliers, duplicates
3. **Feature Engineering** - Extract features from URLs, normalize data
4. **Preprocessing** - Transform data, create train/test splits
5. **Model Training** - Train ML models with hyperparameter tuning
6. **Evaluation** - Validate model performance
7. **Deployment** - Serve model via API (FastAPI)

---

## 📊 Model Performance

*Coming soon - Model evaluation metrics will be added after training phase*

---

## 🤝 Contributing

### Development Workflow
1. Create a feature branch
2. Make your changes
3. **Run tests:** `python3 -m pytest -v`
4. Ensure all tests pass
5. Submit a pull request

### Code Quality Standards
- ✅ All new code must include tests
- ✅ Maintain >80% code coverage
- ✅ Follow PEP 8 style guidelines
- ✅ Document functions and classes

---

## 📝 Project Status

| Phase | Status | Completion |
|-------|--------|------------|
| 1. Data Collection | ✅ Complete | 100% |
| 2. Data Exploration | ✅ Complete | 100% |
| 3. Data Cleaning | ✅ Complete | 100% |
| 4. Feature Engineering | ✅ Complete | 100% |
| 5. Model Training | ✅ Complete | 100% |
| 6. Testing & Validation | ✅ Complete | 100% |
| 7. Model Deployment | ✅ Complete | 100% |
| 8. Monitoring & Maintenance | ⏳ Planned | 0% |

---

## 📚 Documentation

- **[TESTING_README.md](TESTING_README.md)** - Complete testing guide
- **[GUIA_ACTIVACION_TESTS.md](GUIA_ACTIVACION_TESTS.md)** - Test activation guide
- **[RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md)** - Testing executive summary
- **/docs/** - Additional project documentation

---

## 🐛 Known Issues & Limitations

*None reported at this time*

---

## 📧 Contact

For questions or issues, please contact the team members listed in the Team Organization section.

---

## 🙏 Acknowledgments

- Mashable for providing the dataset
- Course instructors and teaching assistants
- Open-source community for MLOps tools

---

**Last Updated:** November 2025  
**Version:** 1.3.0  
**Project Status:** 🚀 Active Development