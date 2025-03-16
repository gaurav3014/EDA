# corn-mycotoxin-prediction
# Machine Learning Pipeline for Mycotoxin Prediction

## Overview
This project aims to analyze corn data and predict DON (vomitoxin) concentration using machine learning techniques. It includes data preprocessing, dimensionality reduction, model training, and evaluation, along with a Streamlit web application for visualization.

## Project Structure
```
├── app.py          # Streamlit web app
├── pipeline.py     # Machine Learning pipeline
├── unit_tests.py   # Unit tests for the pipeline
├── MLE-Assignment.csv # Sample dataset
├── README.md       # Project documentation
```

## Installation
Ensure you have Python 3.8+ installed. Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline
To execute the machine learning pipeline on a dataset:

```bash
python pipeline.py /path/to/MLE-Assignment.csv
```

## Running Unit Tests
To validate the pipeline's correctness, run:

```bash
python unit_tests.py
```

## Running the Streamlit App
Launch the interactive web application with:

```bash
streamlit run app.py
```

## Model Performance
- **Mean Absolute Error (MAE)**: 0.20
- **Root Mean Squared Error (RMSE)**: 0.49
- **R² Score**: 0.85

## Features
- **Data Preprocessing**: Handling missing values, standardization
- **Dimensionality Reduction**: PCA for feature reduction
- **Model Training**: CNN-based regression model
- **Evaluation Metrics**: MAE, RMSE, R² Score
- **Streamlit App**: Visualization of predictions
- **Unit Testing**: Ensuring pipeline correctness

## Contribution
Feel free to fork the repository and submit a pull request for improvements!

## License
MIT License
