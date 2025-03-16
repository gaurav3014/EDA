# Machine Learning Pipeline for Mycotoxin Prediction

## Overview
This repository contains a machine learning pipeline for analyzing corn data and predicting DON concentration levels. The pipeline includes data preprocessing, dimensionality reduction using PCA, training a CNN model, and evaluation. Additionally, a Streamlit app is provided for interactive visualization and inference.

## Project Structure

```
├── app.py             # Streamlit app for visualization and prediction
├── pipeline.py        # Machine Learning pipeline
├── unit_tests.py      # Unit tests for pipeline
├── MLE-Assignment.csv # Sample dataset
├── notebook.ipynb     # Jupyter Notebook for exploratory analysis
├── README.md          # Project documentation
```

## Dataset
- **File**: `MLE-Assignment.csv`
- **Shape**: 5 rows × 450 columns
- **Data Type**: Numerical

## Steps in the Pipeline

### 1. Data Exploration and Preprocessing
- Load dataset, check for missing values and outliers.
- Standardize the data using `StandardScaler`.
- Visualize the data (reflectance plots, histograms, heatmaps).

### 2. Dimensionality Reduction
- Apply **Principal Component Analysis (PCA)** to reduce dimensions.
- Transform the data for model input.

### 3. Model Training
- **Split the dataset** (80% training, 20% testing).
- **Reshape data** for CNN input (samples, time steps, features).
- **Train a CNN model** for DON concentration prediction.
- **Optimize hyperparameters** (Grid Search, Bayesian Optimization).

### 4. Model Evaluation
- **Metrics**:
  - Mean Absolute Error (MAE): **0.20**
  - Root Mean Squared Error (RMSE): **0.49**
  - R² Score: **0.85**
- **Visual Evaluation**:
  - Scatter plots for actual vs predicted values.
  - Residual analysis.
- **Feature Importance Analysis** using SHAP.



## Running the Pipeline

### Run the Full Pipeline
```bash
python pipeline.py /path/to/MLE-Assignment.csv
```

### Run the Pipeline with Unit Tests
```bash
python unit_tests.py
```

### Run the Streamlit App
```bash
streamlit run app.py
```

## Deliverables
- **Jupyter Notebook** (`notebook.ipynb`): Contains exploratory analysis and visualizations.
- **Pipeline Script** (`pipeline.py`): End-to-end ML pipeline.
- **Streamlit App** (`app.py`): Interactive UI.
- **Unit Tests** (`unit_tests.py`): Validates pipeline functionality.


