# ML-Approaches-for-Early-Cardiac-Disease-Prediction 


This project explores early cardiovascular disease (CVD) prediction using two complementary data sources:

Tabular clinical data (70,000 patient records with 12 medical and lifestyle indicators)

Cardiac MRI image data (ACDC dataset with 150 patient MRI volumes)

The study compares classical Machine Learning models for clinical data and a 3D Convolutional Neural Network (3D-CNN) for MRI data to evaluate how different modalities contribute to accurate CVD prediction.

# Team Members

Garima Patwal – H00509869

Mitalikumari Pradipbhai Patel – H00522099

Huda Shaikh – H00506370

Syeda Urooj Fathima – H00511143

Seema Aashikab Anees – H00521785

# Project Overview

This study aims to:

Build and evaluate four machine learning models (Naive Bayes, Logistic Regression, KNN, XGBoost) for clinical CVD prediction.

Develop a 3D-CNN model for automated cardiac classification from MRI volumes.

Compare results and highlight how imaging and clinical data provide complementary insights.

Identify model limitations and propose future improvements for multimodal diagnostic systems.

# Datasets Used 
1. Clinical Tabular Dataset – Kaggle CVD Dataset

150 patient cine-MRI volumes (ED & ES phases)

5 diagnostic categories (Normal, DCM, HCM, MINF, ARV)

Preprocessing:

Resampled to 128 × 128 × 8

Z-score intensity normalization

Label encoding

Two-channel MRI tensor (ED + ES)

Train/val/test split at patient level (80/20 + 50 independent test cases)

2. Clinical Tabular Dataset – Kaggle CVD Dataset

70,000 patient records

12 features: age, BP, cholesterol, glucose, BMI, lifestyle habits

Cleaning & preparation:

Age converted from days → years

Categorical values encoded

BMI engineered from height & weight

Outlier removal for BP

Full univariate, bivariate, and multivariate analysis

# Models Implemented
A. Machine Learning on Clinical Data
| Model                   | Notes                                                            |
| ----------------------- | ---------------------------------------------------------------- |
| **Naive Bayes**         | Baseline probabilistic model; limited by feature independence    |
| **Logistic Regression** | Interpretable linear classifier; stable generalisation           |
| **KNN (k=5)**           | Good at capturing local patterns; slight overfitting             |
| **XGBoost**             | Best-performing model; strong handling of nonlinear interactions |


B. Deep Learning on MRI Data

A 3D-CNN designed for volumetric classification:

4 convolutional blocks with BatchNorm + MaxPool

AdaptiveAvgPool3D for global spatial compression

Fully connected classifier with dropout

Trained for 100 epochs using:

AdamW optimizer

Cosine Annealing scheduler

Cross-Entropy Loss with label smoothing

# Results Summary
MRI Classification – 3D-CNN

Training Accuracy: 83.45%

Validation Accuracy: 76.65%

Test Accuracy: 76.60%

Strengths:

High correctness for Normal and ARV classes

Balanced precision, recall & F1 across all five classes

Limitations:

Moderate overfitting

Overlap between cardiomyopathy subtypes (DCM–HCM)

# Clinical Data Classification

| Model                   | Performance Summary                                                        |
| ----------------------- | -------------------------------------------------------------------------- |
| **Naive Bayes**         | Weakest; high false negatives                                              |
| **Logistic Regression** | Balanced and stable                                                        |
| **KNN**                 | Good detection but overpredicts positive cases                             |
| **XGBoost**             | ⭐ **Best model (~88% accuracy)**; high precision & recall for both classes |

# Visualizations Included

The project provides the following:

Epoch vs Accuracy (3D-CNN)

Epoch vs Loss (3D-CNN)

Confusion matrices for all models

Precision–Recall–F1 charts

Clinical dataset distribution and correlation visuals

# Key Findings

Clinical features such as blood pressure, BMI, cholesterol, glucose, age remain highly predictive of CVD.

XGBoost is the most suitable model for low-cost, large-scale screening.

MRI-based 3D-CNN adds value in distinguishing structural cardiac abnormalities.

Clinical and imaging data provide complementary strengths—ideal for multimodal diagnostic systems.

Dataset constraints (size, diversity, single-modality imaging) affect generalizability.

# Future Work

Combine MRI + clinical data using multimodal fusion

Improve MRI classification via:

Data augmentation

Multi-centre dataset expansion

Feature extraction from segmentation masks

Add explainability (Grad-CAM, SHAP)

Apply uncertainty estimation for safer clinical use

# Technologies Used

Python, NumPy, Pandas, Scikit-Learn

PyTorch for 3D-CNN

Matplotlib / Seaborn for visualisation

XGBoost library

GPU acceleration for deep learning



