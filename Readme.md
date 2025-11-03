# 🧠 MLOps Project – Predicting Article Popularity on Social Networks

## 📋 Project Overview
This project aims to **predict the number of social media shares** of online articles published by **Mashable** over a two-year period.  
The dataset **summarizes a heterogeneous set of features** about each article — including textual, temporal, and contextual metrics — to estimate its potential **popularity**.

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
└── README.md                # Main project documentation
```

> 🧭 **Note:** This structure supports reproducibility and scalability with tools like **DVC**, **MLflow**, and **Docker**.

---
