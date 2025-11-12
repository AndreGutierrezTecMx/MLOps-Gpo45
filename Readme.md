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

## 🔧 MLOps Tools & Technologies

| Tool | Purpose | Status |
|------|---------|--------|
| **DVC** | Data Version Control | ✅ Implemented |
| **MLflow** | Experiment Tracking | ✅ Implemented |
| **pytest** | Automated Testing | ✅ Implemented |
| **scikit-learn** | ML Modeling | ✅ Implemented |
| **pandas** | Data Processing | ✅ Implemented |
| **XGBoost** | Gradient Boosting | ✅ Implemented |
| **Docker** | Containerization | 🔄 In Progress |
| **FastAPI** | Model Serving | 🔄 In Progress |

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
| 5. Model Training | 🔄 In Progress | 75% |
| 6. Testing & Validation | ✅ Complete | 100% |
| 7. Model Deployment | 🔄 In Progress | 25% |
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

## 📄 License

*License information to be added*

---

## 🙏 Acknowledgments

- Mashable for providing the dataset
- Course instructors and teaching assistants
- Open-source community for MLOps tools

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Project Status:** 🚀 Active Development
