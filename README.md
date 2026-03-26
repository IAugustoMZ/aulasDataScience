# Data Science and Machine Learning Lessons

Welcome to this comprehensive repository! It contains Data Science, Machine Learning, and MLOps materials developed for classroom lessons and professional practice. The goal is to provide practical, accessible learning material covering fundamentals of Machine Learning, advanced Deep Learning techniques, and production-grade MLOps pipelines.

## Repository Structure

The repository is organized into **6 thematic modules**, arranged for a logical learning progression:

| Module | Topic | Content |
|--------|-------|---------|
| [ML Fundamentals](#1-machine-learning-fundamentals) | Introduction to scikit-learn and ML workflow | 6 notebooks |
| [Classification](#2-classification-techniques) | Logistic Regression, Decision Trees, SVM | 13 notebooks |
| [Clustering](#3-clustering-techniques) | KMeans, Hierarchical Clustering, DBSCAN | 6 notebooks |
| [Neural Networks](#4-neural-networks-with-tensorflow) | ANN for classification, regression, and time series | 4 notebooks |
| [Deep Learning](#5-deep-learning-with-tensorflow) | CNN, Transfer Learning, LSTM, Autoencoders, SOM | 5 notebooks |
| [MLOps & Model Operationalization](#6-mlops--model-operationalization) | Production pipelines, pipelines, data validation, and model deployment | 2 aulas + 2 reference implementations |

**Total: 35 Jupyter notebooks + 4 production Python scripts** covering theory, practical implementation, applied projects, and MLOps best practices.

### Note on Student Exercise Files (_alunos versions)
This repository includes paired notebook versions:
- **Regular notebooks** (e.g., `1_introducao_classificacao.ipynb`): Complete solutions and guides
- **_alunos versions** (e.g., `1_introducao_classificacao_alunos.ipynb`): Student exercise templates for classroom use

---

## 1. Machine Learning Fundamentals

An introductory module that establishes the core concepts of supervised machine learning using scikit-learn. It covers the full ML workflow: data loading, exploratory analysis, training, prediction, and performance evaluation.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [ML Fundamentals with Scikit-Learn 01](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn01_alunos.ipynb) | Practical introduction to Logistic Regression and K-Nearest Neighbors (KNN) using the Iris and Credit Scoring datasets. Covers train-test split, accuracy score, and algorithm comparison. |
| 2 | [ML Fundamentals with Scikit-Learn 02](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn02_alunos.ipynb) | Classification model selection and evaluation techniques. |
| 3 | [ML Fundamentals with Scikit-Learn 03](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn03_alunos.ipynb) | Feature scaling and preprocessing strategies. |
| 4 | [ML Fundamentals with Scikit-Learn 04](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn04_alunos.ipynb) | Model validation and performance metrics. |
| 6 | [ML Fundamentals with Scikit-Learn 06](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn06.ipynb) | Advanced preprocessing and pipeline construction. |
| 7 | [ML Fundamentals with Scikit-Learn 07](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn07_alunos.ipynb) | Practical applications and capstone exercises. |

**Libraries:** scikit-learn, pandas, seaborn, matplotlib

---

## 2. Classification Techniques

A comprehensive module on supervised classification that progresses from simple, interpretable models to advanced techniques like SVM. Emphasizes that model complexity does not guarantee better results—proper evaluation and validation are essential.

### Lesson Notebooks (with Student Versions Available)

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Introduction to Classification](classificacao/1_introducao_classificacao.ipynb) | ML fundamentals: learning types (supervised, unsupervised, semi-supervised, reinforcement), bias-variance tradeoff, linear separability, and evaluation metrics (accuracy, precision, recall, F1-score, AUC-ROC). Intro to Logistic Regression with the sigmoid function on the Iris dataset. |
| 2 | [Logistic Regression](classificacao/2_regressao_logistica.ipynb) | Practical Logistic Regression on the German Credit dataset (1,000 samples, 20 features). Covers preprocessing pipelines with ColumnTransformer, scaling strategies (MinMaxScaler), coefficient interpretation, and business-oriented cost functions. |
| 3 | [Decision Boundary Analysis](classificacao/3_analise_fronteira_decisao.ipynb) | Decision threshold optimization and probability calibration. Demonstrates that the default 0.5 threshold may not be optimal for business problems. Covers ROC and Precision-Recall curves, AUC, and cost-matrix-driven profit optimization. |
| 4 | [Decision Trees & Cross-Validation](classificacao/4_arvores_de_decisao_validacao_cruzada.ipynb) | Decision Trees with Gini and Entropy criteria, demonstration of overfitting, hyperparameter tuning (max_depth), Stratified K-Fold Cross-Validation (k=10), GridSearchCV, custom metrics, and handling class imbalance with class_weight. Dataset: Drug200. |
| 5 | [Support Vector Machines (SVM)](classificacao/5_maquinas_suportadas_vetor.ipynb) | Maximum-margin classifiers with kernels (Linear, RBF, Sigmoid, Cosine). Covers regularization parameter C, gamma, RandomizedSearchCV for efficient tuning, and SVM advantages in high-dimensional spaces. Dataset: German Credit. |

### Projects & Solutions

| # | Notebook | Description |
|---|----------|-------------|
| 6 | [Data Science Project - Classification](classificacao/projeto_data_science.ipynb) | End-to-end project on the Drug200 dataset (multiclass classification of 5 drugs). Includes full EDA, preprocessing with OneHotEncoder and RobustScaler, One-vs-All strategy, and comparison of Logistic Regression, Decision Trees, and SVM with cross-validation. Best result: 97% accuracy with Decision Trees. |
| 7 | [Solutions - Logistic Regression Exercises](classificacao/CorreçãoExercíciosRegressãoLogística.ipynb) | Solutions focused on imbalanced data (73% vs 27%). Compares scaling strategies (MinMaxScaler, RobustScaler, StandardScaler), implements custom cost functions (TP=+100, FP=-150, TN=+100, FN=-250), and break-even threshold analysis. |
| 8 | [Solutions - Cross-Validation Exercises](classificacao/CorreçãoExercícios_ValidacaoCruzada_alunos.ipynb) | Advanced application with categorical and numerical features. Shows that adding categorical variables improved accuracy from 72% to 97%. Compares Decision Trees (F1: 0.51) vs Logistic Regression (F1: 0.65) using business-aware metrics. |

**Libraries:** scikit-learn, pandas, numpy, seaborn, matplotlib
**Datasets:** Iris, German Credit (1,000 samples), Drug200 (200 samples)

---

## 3. Clustering Techniques

Module on unsupervised learning and clustering techniques. Progresses from distance-based algorithms (KMeans), through hierarchical methods, to density-based algorithms (DBSCAN), with applied projects on real data.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Clustering Case Study](clusterizacao/1_estudo_caso_agrupamento.ipynb) | Intro to unsupervised learning with KMeans on the German Credit dataset (1,000 records, 21 features). Covers Silhouette Score, Homogeneity, Completeness, and V-Score, using pipeline-based modeling. |
| 2 | [KMeans](clusterizacao/2_KMeans.ipynb) | In-depth KMeans exploration: algorithm mechanics, k-means++ initialization, elbow method, and silhouette analysis for selecting k. Demonstrates the impact of data standardization on Iris and German Credit datasets. |
| 3 | [Mini Clustering Project](clusterizacao/3_mini-projeto-clusterizacao.ipynb) | Customer segmentation for an e-commerce dataset with 541,909 transactions. Includes temporal feature engineering, comparison of KMeans, Hierarchical Clustering, and DBSCAN, and evaluation using silhouette and Gap Statistics. |
| 4 | [Hierarchical Clustering](clusterizacao/4_clusterizacao_hierarquica.ipynb) | Detailed analysis of Agglomerative Hierarchical Clustering with linkage strategies (single, complete, average, centroid, Ward), distance metrics (Euclidean, Manhattan, Minkowski), dendrogram visualization, and comparison with KMeans. |
| 5 | [DBSCAN](clusterizacao/5_dbscan.ipynb) | Introduction to DBSCAN (Density-Based Spatial Clustering of Applications with Noise). Demonstrates automatic cluster discovery and outlier handling, with sensitivity analysis for eps and min_samples on the Iris dataset. |
| 6 | [Project - Comparing Clustering Techniques](clusterizacao/6_projeto_comparacao_clusterizacao.ipynb) | Comparative project between KMeans, Hierarchical Clustering (4 linkages), and DBSCAN on the Garment Worker Productivity dataset (1,197 records, 15 variables). Includes missing value handling, outlier removal, log transforms, and cluster profiling. |

**Libraries:** scikit-learn, pandas, numpy, seaborn, matplotlib, scipy
**Datasets:** German Credit, Iris, E-commerce (541,909 transactions), Garment Worker Productivity (1,197 records)

---

## 4. Neural Networks with TensorFlow

Practical module on artificial neural networks (ANN) covering the three main ML problem types: classification, regression, and time series. Uses TensorFlow/Keras as the primary framework.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Neural Networks - Fundamentals](redes_neurais/1_redes_neurais.ipynb) | Theoretical and practical introduction: single-layer perceptron and multilayer perceptron (MLP), activation functions (sigmoid, ReLU, tanh, softmax), forward/backward propagation, backpropagation algorithm, and gradient-based optimization. |
| 2 | [ANN Project - Classification](redes_neurais/2_projeto_ann_classificacao.ipynb) | Practical binary and multiclass classification with dense neural networks. Covers architecture design, loss functions (binary/categorical cross-entropy), regularization strategies, ROC-AUC analysis, and hyperparameter tuning with Adam and SGD optimizers. |
| 3 | [ANN Project - Regression](redes_neurais/3_projeto_ann_regressao.ipynb) | Neural networks for continuous value prediction. Covers MSE and MAE losses, Dropout and L1/L2 regularization, residual analysis, R-squared, and output layer activation choices (Linear, ReLU). |
| 4 | [Neural Networks for Time Series](redes_neurais/4_redes_neurais_series_temporais.ipynb) | Time series forecasting with LSTM and GRU. Covers sequence preparation with sliding windows, bidirectional architectures, stationarity analysis (ACF/PACF), TimeseriesGenerator, and early stopping with callbacks. Applied to Bitcoin price prediction. |

**Libraries:** TensorFlow/Keras, scikit-learn, pandas, numpy, matplotlib
**Datasets:** Iris, cancer classification data, housing prices, Bitcoin prices

---

## 5. Deep Learning with TensorFlow

Advanced module covering specialized Deep Learning architectures: convolutional networks for images, transfer learning, recurrent networks for time series, autoencoders, and hybrid models.

| # | Notebook | Description |
|---|----------|-------------|
| 0 | [SOM + ANN - Fraud Detection](deep_learning/0_SOM%2BANN.ipynb) | Hybrid model combining Self-Organizing Maps (SOM) for unsupervised anomaly detection with an ANN (MLP) for supervised fraud classification. The SOM identifies anomalous transactions and the ANN classifies fraud. Dataset: 690 records; ~90% accuracy. |
| 1 | [Convolutional Neural Networks (CNN)](deep_learning/1_redes_neurais_convolucionais.ipynb) | From-scratch CNNs for binary image classification (cats vs dogs). Covers convolutional layers with ReLU, max pooling, data augmentation (rescaling, shearing, zoom, flip), and architecture with Flatten and Dense layers. Training dataset: ~8,000 images, achieving 78–80% accuracy. |
| 2 | [Transfer Learning](deep_learning/2_transfer_learning.ipynb) | Transfer Learning using VGG16 (pretrained on ImageNet) as a feature extractor with frozen weights. Custom Dense layers for binary classification. Demonstrates superior performance over a CNN trained from scratch: 91.7% vs 80% on the same cats vs dogs dataset. |
| 3 | [RNN and LSTM](deep_learning/3_rnn_lstm.ipynb) | Google stock price prediction with stacked LSTM (200-100-50-25 units). Uses a 5-day sliding window, Dropout (0.2), MinMaxScaler normalization, and MSE optimization. Dataset: 1,258 daily records (Jan/2012 - Jul/2015). |
| 4 | [Variational Autoencoders (VAE)](deep_learning/4_autoencoders.ipynb) | Variational Autoencoder with a convolutional encoder and transposed-convolution decoder for MNIST image compression into a 2D latent space. Covers the reparameterization trick, combined KL-divergence + binary cross-entropy loss, latent-space digit clustering, and smooth latent interpolations. |

**Libraries:** TensorFlow/Keras, MiniSom, scikit-learn, pandas, numpy, matplotlib, scipy
**Datasets:** Credit Card Applications (690 records), Dogs vs Cats (10,000 images), Google Stock Prices (1,258 records), MNIST (70,000 images)

---

---

## 6. MLOps — Model Operationalization

Comprehensive module on MLOps engineering, covering the full lifecycle from problem definition to production deployment. Progresses from foundational concepts through a complete, reference-grade production pipeline.

### Educational Content

| # | Content | Description |
|---|---------|-------------|
| 1 | [Aula 01 — Intro to MLOps](operacionalizacao_modelos_mlops/aula01/operacionalizacao_modelos_mlops_aula01.ipynb) | **Conceptual foundation** — Why ML projects fail in production, the executor-to-engineer mindset shift, and the role of operationalization in the ML lifecycle. Sets context for production-grade practices. |

### Production-Grade Pipeline Implementation

| # | Name | Type | Description |
|---|------|------|-------------|
| 2 | [Aula 02 — Full MLOps Pipeline with EDA & Modeling](operacionalizacao_modelos_mlops/aula02/) | **Complete Pipeline** | **Production-ready implementation** with 5-stage architecture: (1) Data Ingestion (CSV → Parquet via Kaggle API), (2) EDA (7-module analysis), (3) Data Quality (Great Expectations validation), (4) Preprocessing (9 sklearn transformers), (5) Modeling (10 base models + 2 ensembles with Optuna + MLFlow). **Dataset:** California Housing (Kaggle, 20,640 rows). |

**Aula02 Architecture:**
```
Step 1: INGESTION
├─ Kaggle API → Download CSV
├─ PyArrow Streaming → Parquet conversion
└─ Lazy loading for large datasets

Step 2: EDA (7 modules)
├─ Descriptive Statistics
├─ Visualizations & Distributions
├─ Pivot Tables & Aggregations
├─ Statistical Tests & Correlations
├─ Interaction Effects & Feature Engineering
├─ Clustering Analysis
└─ Comprehensive report → outputs/

Step 3: DATA QUALITY
├─ Great Expectations suite with custom rules
├─ YAML-driven validation profiles
└─ Error tracking & reporting

Step 4: PREPROCESSING
├─ 9 sklearn Transformers (StandardScaler, OneHotEncoder, etc.)
├─ YAML configuration for reproducibility
└─ Fitted transformer serialization

Step 5: MODELING
├─ 10 base models (Linear, Ridge, Lasso, RF, GB, SVR, KNN, ElasticNet, AdaBoost, Extra Trees)
├─ 2 ensemble models (Voting, Stacking)
├─ Hyperparameter tuning (Optuna)
├─ MLFlow experiment tracking
└─ Model selection & comparison
```

**Aula02 File Structure:**
```
aula02/
├── config/              (5 YAML files controlling all behavior)
├── eda/                 (7 Python modules for analysis)
├── notebooks/           (4 walkthrough Jupyter scripts)
│   ├── 1_ingestao.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_qualidade.ipynb
│   └── 4_preprocessing.ipynb
├── src/                 (Production utilities)
│   ├── downloader.py
│   ├── ingestion.py
│   ├── quality.py
│   └── preprocessing.py
├── main.py              (CLI orchestrator)
├── outputs/             (Results: 33 figs, 10 tables, quality reports)
└── README.md            (Full documentation)
```

**Key Aula02 Findings:**
- `median_income` strongest predictor (Pearson r=0.688)
- `ocean_proximity` explains 23.8% of variance; INLAND ~$115k cheaper
- Engineered features (`bedrooms_per_room`, `rooms_per_household`) outperform raw counts
- Geographic distance from major cities: r=-0.384 with house value
- 3 geographic housing market clusters: LA Basin · Bay Area · Affluent Coastal
- **Recommendation:** log-transform target; use income + location + ratio features

**Libraries:** PyArrow, Great Expectations, Optuna, MLFlow, scikit-learn, pandas, numpy, scipy, matplotlib, seaborn

---

### Reference Implementations

| # | Name | Type | Purpose |
|---|------|------|---------|
| 3 | [Demo Project](operacionalizacao_modelos_mlops/demo_projeto/) | **Scaffold** | Simplified MLOps project template for quick prototyping. Clean folder structure (`config/`, `data/`, `notebooks/`, `outputs/`, `src/`) ready for customization. |
| 4 | [Reference Project](operacionalizacao_modelos_mlops/ref_projeto/) | **Production Reference** | Full-featured MLOps implementation with production patterns. Includes multi-domain `config/`, complete pipeline from ingestion → modeling, walkthrough scripts for each stage with `# %%` cells for interactive execution, and MLFlow tracking (`mlruns/`). |

**Reference Project Walkthroughs:**
```
ref_projeto/
├── ingestao_walkthrough.py       (Data loading & preparation)
├── qualidade_walkthrough.py      (Data validation & profiling)
├── preprocessamento_walkthrough.py (Transformation pipeline)
├── modelagem_walkthrough.py      (Model training & evaluation)
└── mlruns/                       (MLFlow experiment history)
```

**Libraries (MLOps Core):** PyArrow, PyYAML, Great Expectations, Optuna, MLFlow
**Libraries (EDA/Preprocessing):** scikit-learn, pandas, numpy, scipy, statsmodels, matplotlib, seaborn
**Datasets:** California Housing (Kaggle — 20,640 rows, 10 features)

---

## Technologies & Dependencies

### Core Stack
- **Python 3.11** — Programming language
- **scikit-learn** — Classical ML algorithms, preprocessing, metrics
- **TensorFlow / Keras** — Neural networks and Deep Learning
- **pandas / numpy** — Data manipulation and processing
- **matplotlib / seaborn** — Data visualization
- **scipy** — Scientific computing and advanced statistics
- **MiniSom** — Self-Organizing Maps

### MLOps & Production
- **PyArrow** — Efficient data serialization (CSV ↔ Parquet)
- **PyYAML** — Configuration management
- **Great Expectations** — Data validation and profiling
- **Optuna** — Hyperparameter optimization
- **MLFlow** — Experiment tracking and model registry
- **Kaggle API** — Dataset downloads
- **FastAPI / Flask** — Model serving (in reference implementations)

### Environment
- **Virtual Environment:** `ds_env/` (Python 3.11 venv)
- **Jupyter Ecosystem:** For interactive development and education

---

## Quick Start Guide

### 1. Clone & Activate Environment

```bash
git clone https://github.com/[your-username]/aulasDataScience.git
cd aulasDataScience

# Activate virtual environment (Windows)
ds_env\Scripts\activate

# Activate virtual environment (macOS/Linux)
source ds_env/bin/activate
```

### 2. Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks

**For individual lessons:**
```bash
jupyter notebook classificacao/1_introducao_classificacao.ipynb
```

**To browse all notebooks:**
```bash
jupyter notebook
```

### 4. Run MLOps Pipeline (Aula02)

```bash
cd operacionalizacao_modelos_mlops/aula02

# Run full pipeline
python main.py --all

# Or run individual stages
python main.py --ingestion
python main.py --eda
python main.py --quality
python main.py --preprocessing
python main.py --modeling
```

### 5. Run Reference Implementation Walkthroughs

```bash
cd operacionalizacao_modelos_mlops/ref_projeto

# Each walkthrough uses # %% cells for interactive execution in VS Code / Jupyter:
python ingestao_walkthrough.py
python qualidade_walkthrough.py
python preprocessamento_walkthrough.py
python modelagem_walkthrough.py
```

---

## Project Organization & Learner Path

### For Beginners
**Start Here:** `fundamentos_machine_learn/` → `classificacao/1_introducao_classificacao.ipynb`

- Learn ML fundamentals, train-test splits, and basic algorithms
- Understand evaluation metrics before diving into complex models
- Use the paired `_alunos` versions for exercises

### For Intermediate Learners
**Progress To:** `classificacao/` and `clusterizacao/`

- Explore multiple algorithm families (Logistic Regression, Decision Trees, SVM, Clustering)
- Practice hyperparameter tuning and cross-validation
- Work through projects with real datasets

### For Advanced Learners
**Move To:** `redes_neurais/` and `deep_learning/`

- Build neural networks from scratch
- Understand backpropagation and modern architectures (CNN, LSTM, VAE)
- Apply transfer learning to image classification

### For MLOps & Production
**Explore:** `operacionalizacao_modelos_mlops/`

- **Aula01:** Conceptual grounding in MLOps principles
- **Aula02:** Hands-on production pipeline with all 5 stages
- **Reference Projects:** Production-ready templates and best practices

---

## Datasets & Data Files

This repository uses publicly available datasets for educational purposes:

| Dataset | Location | Size | Notebooks Using It |
|---------|----------|------|-------------------|
| **Iris** | Built-in (seaborn/sklearn) | 150 samples | Fundamentals, Classification, Clustering, Neural Networks |
| **German Credit** | `classificacao/`, `data/` | 1,000 samples | Classification, Clustering |
| **Drug200** | `classificacao/drug200.csv` | 200 samples | Classification (with multiclass) |
| **Credit Scoring** | `data/cs-training (1).csv` | ~5,000 samples | Advanced preprocessing |
| **E-commerce** | Generated in notebooks | 541,909 transactions | Clustering (mini-project) |
| **California Housing** | Kaggle (auto-download in Aula02) | 20,640 samples | MLOps pipeline |
| **Garment Worker Productivity** | Generated in notebooks | 1,197 records | Clustering comparison |
| **CO2 Emissions** | `data/co2.csv` | Various | Time series / auxiliary |

**Note:** Datasets are either built-in (Iris), small CSV files included, or downloaded automatically during pipeline execution. No API keys required except for Kaggle (optional for Aula02).

---

## Repository Status & Content Summary

```
Total Content Inventory:
├── Jupyter Notebooks:           35 (lesson content + student exercises)
├── Python Scripts:              4 (MLOps walkthroughs in ref_projeto/)
├── Data Files (CSV):            6
├── Configuration Files (YAML):  5 (Aula02 MLOps config)
├── Documentation Files:         4 (README, CLAUDE.md, Aula02/README, outputs/)
└── Python Virtual Environment:  Complete (150+ packages)

Learning Modules:               6 (Fundamentals → MLOps)
Total Estimated Content Hours:  40-50 (including deep dives)
Student Exercise Templates:     8 (_alunos versions in Fundamentals & Classification)
Production Implementations:     2 (demo_projeto, ref_projeto)
```

---

## Tips for Using This Repository

### As an Instructor
- **For Lectures:** Use the regular notebooks (`1_introducao.ipynb`, etc.)
- **For Assignments:** Direct students to the `_alunos` versions for hands-on practice
- **For Projects:** Assign one from the Projects & Exercises sections in each module
- **For Demonstrations:** Use MLOps Aula02 to show real production workflows

### As a Student
- **Read First:** Start with the theory sections (Aula01 for conceptual grounding)
- **Code Along:** Open both regular and `_alunos` notebooks side-by-side; try exercises independently first
- **Projects:** Build one project per module to solidify understanding
- **Production Ready:** Explore Aula02 and reference implementations to understand industry practices

### For Using Your Own Data
1. Place datasets in the `data/` folder
2. Update notebook paths or configuration files (YAML in MLOps pipeline)
3. Adapt preprocessing and EDA modules to your domain
4. Reuse the model comparison framework from projects

---

## License

This repository is provided for **educational purposes**. Feel free to use, modify, and adapt the notebooks for learning and teaching.

### Datasets
Some datasets are publicly available (Iris, Housing); others are from Kaggle or open sources. Respect the licensing terms of individual datasets when used.

---

## Contributing & Feedback

Have suggestions for improvements, found an error, or want to add content?
- Open an issue or pull request
- This repository is a living educational resource and improvements are always welcome

---

## Project Philosophy

This repository embodies three core principles:

1. **Learning-Focused:** Every notebook prioritizes clarity and understanding over production optimization
2. **Practical:** Theory is always paired with real code and datasets; no "toy examples" in isolation
3. **Progressive:** Content builds from fundamentals through advanced techniques to professional MLOps practices

Our goal: Transform from "executor" (running code provided by others) to "engineer" (understanding, designing, and deploying ML systems).
