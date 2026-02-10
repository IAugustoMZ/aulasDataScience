# Data Science and Machine Learning Lessons

Welcome to this repository! It contains Data Science and Machine Learning notebooks developed for classroom lessons. The goal is to share practical, accessible learning material, covering fundamentals of Machine Learning up to advanced Deep Learning techniques.

## Repository Structure

The repository is organized into **5 thematic modules**, arranged for a logical learning progression:

| Module | Topic | Notebooks |
|--------|-------|-----------|
| [ML Fundamentals](#1-machine-learning-fundamentals) | Introduction to scikit-learn and ML workflow | 1 |
| [Classification](#2-classification-techniques) | Logistic Regression, Decision Trees, SVM | 8 |
| [Clustering](#3-clustering-techniques) | KMeans, Hierarchical Clustering, DBSCAN | 6 |
| [Neural Networks](#4-neural-networks-with-tensorflow) | ANN for classification, regression, and time series | 4 |
| [Deep Learning](#5-deep-learning-with-tensorflow) | CNN, Transfer Learning, LSTM, Autoencoders, SOM | 5 |

**Total: 24 notebooks** covering theory, practical implementation, and applied projects.

---

## 1. Machine Learning Fundamentals

An introductory module that establishes the core concepts of supervised machine learning using scikit-learn. It covers the full ML workflow: data loading, exploratory analysis, training, prediction, and performance evaluation.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [ML Fundamentals with Scikit-Learn](fundamentos_machine_learn/fundamentos_machine_learning_scikitlearn01_alunos.ipynb) | Practical introduction to Logistic Regression and K-Nearest Neighbors (KNN) using the Iris and Credit Scoring datasets. Covers train-test split, accuracy score, and algorithm comparison. |

**Libraries:** scikit-learn, pandas, seaborn, matplotlib

---

## 2. Classification Techniques

A comprehensive module on supervised classification that progresses from simple, interpretable models to advanced techniques like SVM. Emphasizes that model complexity does not guarantee better results—proper evaluation and validation are essential.

### Theory and Practical Lessons

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Introduction to Classification](classificacao/1_introducao_classificacao.ipynb) | ML fundamentals: learning types (supervised, unsupervised, semi-supervised, reinforcement), bias-variance tradeoff, linear separability, and evaluation metrics (accuracy, precision, recall, F1-score, AUC-ROC). Intro to Logistic Regression with the sigmoid function on the Iris dataset. |
| 2 | [Logistic Regression](classificacao/2_regressao_logistica.ipynb) | Practical Logistic Regression on the German Credit dataset (1,000 samples, 20 features). Covers preprocessing pipelines with ColumnTransformer, scaling strategies (MinMaxScaler), coefficient interpretation, and business-oriented cost functions. |
| 3 | [Decision Boundary Analysis](classificacao/3_analise_fronteira_decisao.ipynb) | Decision threshold optimization and probability calibration. Demonstrates that the default 0.5 threshold may not be optimal for business problems. Covers ROC and Precision-Recall curves, AUC, and cost-matrix-driven profit optimization. |
| 4 | [Decision Trees & Cross-Validation](classificacao/4_arvores_de_decisao_validacao_cruzada.ipynb) | Decision Trees with Gini and Entropy criteria, demonstration of overfitting, hyperparameter tuning (max_depth), Stratified K-Fold Cross-Validation (k=10), GridSearchCV, custom metrics, and handling class imbalance with class_weight. Dataset: Drug200. |
| 5 | [Support Vector Machines (SVM)](classificacao/5_maquinas_suportadas_vetor.ipynb) | Maximum-margin classifiers with kernels (Linear, RBF, Sigmoid, Cosine). Covers regularization parameter C, gamma, RandomizedSearchCV for efficient tuning, and SVM advantages in high-dimensional spaces. Dataset: German Credit. |

### Projects and Exercises

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

## Technologies Used

- **Python 3**
- **scikit-learn** — classical ML algorithms, preprocessing, and metrics
- **TensorFlow / Keras** — neural networks and Deep Learning
- **pandas / numpy** — data manipulation and processing
- **matplotlib / seaborn** — data visualization
- **scipy** — scientific computing and dendrograms
- **MiniSom** — Self-Organizing Maps

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/IAugustoMZ/aulasDataScience.git
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn tensorflow matplotlib seaborn scipy minisom
   ```

3. Open the notebooks with Jupyter Notebook or Google Colab.

## License

This repository is provided for educational purposes.
