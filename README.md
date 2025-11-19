# ML-Approaches-for-Early-Cardiac-Disease-Prediction 


This project explores early cardiovascular disease (CVD) prediction using two complementary data sources:

Tabular clinical data (70,000 patient records with 12 medical and lifestyle indicators)

Cardiac MRI image data (ACDC dataset with 150 patient MRI volumes)

The study compares classical Machine Learning models for clinical data and a 3D Convolutional Neural Network (3D-CNN) for MRI data to evaluate how different modalities contribute to accurate CVD prediction.

# Team Members

Garima Patwal

Mitalikumari Pradipbhai Patel 

Huda Shaikh

Syeda Urooj Fathima

Seema Aashikab Anees

# Project Overview

This study aims to:

Build and evaluate four machine learning models (Naive Bayes, Logistic Regression, KNN, XGBoost) for clinical CVD prediction.

Develop a 3D-CNN model for automated cardiac classification from MRI volumes.

Compare results and highlight how imaging and clinical data provide complementary insights.

Identify model limitations and propose future improvements for multimodal diagnostic systems.

# Datasets Used 
This project uses two complementary datasets: a clinical tabular dataset and a cardiac MRI imaging dataset. The clinical dataset, sourced from Kaggle, contains 70,000 patient records with 12 medical, demographic, and lifestyle attributes, including age, gender, blood pressure, cholesterol, glucose, BMI, smoking habits, alcohol use, and physical activity. These features were collected during routine medical examinations and provide essential risk indicators for cardiovascular disease. Several preprocessing steps were applied, such as converting age from days to years, encoding categorical variables, engineering BMI from height and weight, removing outliers in systolic and diastolic values, and performing univariate, bivariate, and multivariate analyses to understand feature relationships.

The second dataset comes from the ACDC Challenge (Automated Cardiac Diagnosis Challenge), containing 150 cardiac MRI scans grouped into five diagnostic categories—Normal, DCM, HCM, MINF, and ARV. Each patient includes end-diastolic (ED) and end-systolic (ES) cine-MRI volumes, allowing analysis of cardiac function across phases. The dataset is clean, well-balanced, and includes segmentation masks and metadata. MRI volumes were preprocessed through spatial resampling to 128×128×8, z-score normalization, label encoding, and structuring into two-channel 3D tensors for deep learning. Together, these datasets provide both structured clinical indicators and anatomical imaging information, enabling robust machine-learning and deep-learning approaches for cardiovascular disease prediction.


# Models Implemented
A. Machine Learning on Clinical Data
| Model                   | Notes                                                            |
| ----------------------- | ---------------------------------------------------------------- |
| **Naive Bayes**         | Baseline probabilistic model; limited by feature independence    |
| **Logistic Regression** | Interpretable linear classifier; stable generalisation           |
| **KNN (k=5)**           | Good at capturing local patterns; slight overfitting             |
| **XGBoost**             | Best-performing model; strong handling of nonlinear interactions |


B. Deep Learning on MRI Data

 | Component | Description |
|----------|-------------|
| **Model Type** | 3D-CNN for volumetric classification |
| **Convolutional Blocks** | 4 blocks, each with Conv3D → BatchNorm3D → MaxPool3D |
| **Pooling Layer** | AdaptiveAvgPool3D for global spatial compression |
| **Classifier** | Fully Connected layers with Dropout |
| **Training Epochs** | 100 epochs |
| **Optimizer** | AdamW |
| **Learning Rate Scheduler** | Cosine Annealing |
| **Loss Function** | Cross-Entropy Loss with Label Smoothing |


# Results Summary

| **MRI Classification – 3D-CNN** | **Value** |
|----------------------------------|-----------|
| **Training Accuracy**            | 83.45%    |
| **Validation Accuracy**          | 76.65%    |
| **Test Accuracy**                | 76.60%    |


| **Clinical Tabular Data -Model**            | Train Accuracy (%) | Validation Accuracy (%) | Test Accuracy (%) |
|---------------------------------------------|--------------------|-------------------------|--------------------|
| Naive Bayes               | 75              | 70                     | 68              |
| Logistic Regression        | 85              | 82                     | 80              |
| K-Nearest Neighbors (k=5) | 90              | 78                     | 76              |
| XGBoost                    | 95              | 90                     | 88              |



# Clinical Data Classification

| Model                   | Performance Summary                                                        |
| ----------------------- | -------------------------------------------------------------------------- |
| **Naive Bayes**         | Weakest; high false negatives                                              |
| **Logistic Regression** | Balanced and stable                                                        |
| **KNN**                 | Good detection but overpredicts positive cases                             |
| **XGBoost**             | **Best model (~88% accuracy)**; high precision & recall for both classes |

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



