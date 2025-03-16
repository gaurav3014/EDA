# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page title
st.title("Mycotoxin Level Prediction")

# Step 1: Upload CSV file
st.header("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())

    df = data.copy()

    # Step 2: Preprocessing
    st.header("Step 2: Data Preprocessing")
    st.write("### Handling Null Values")
    st.write(df.isnull().sum())

    st.write("### Data Information")
    st.write(df.info())

    df.dropna(inplace=True)

    # Step 3: Splitting the data and performing PCA
    st.header("Step 3: Data Splitting and PCA")
    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(Y.values.reshape(-1, 1)).ravel()

    pca = PCA(n_components=20)
    x_pca = pca.fit_transform(x_scaled)

    # Plot 1: Average Reflectance Over Wavelengths (Line Plot)
    st.write("### Average Reflectance Over Wavelengths")
    plt.figure(figsize=(12, 6))
    plt.plot(X.mean(axis=0))
    plt.title("Average Reflectance Over Wavelengths")
    plt.xlabel("Wavelength Band")
    plt.ylabel("Reflectance")
    st.pyplot(plt)

    # Plot 2: Distribution of DON Concentration (Box Plot)
    st.write("### Distribution of DON Concentration (Box Plot)")
    plt.figure(figsize=(8, 6))
    sns.boxplot(Y)
    plt.title("Distribution of DON Concentration (vomitoxin_ppb)")
    plt.xlabel("DON Concentration")
    st.pyplot(plt)

   

    X_train, X_test, y_train, y_test = train_test_split(x_pca, y_scaled, test_size=0.2, random_state=42)

    # Reshape for CNN input (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Step 4: Building and Training the CNN Model
    st.header("Step 4: Building and Training the CNN Model")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[es])

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Evaluation")
    st.write(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')

    # Plot 4: Actual vs Predicted DON Concentration (Scatter Plot)
    st.write("### Actual vs Predicted DON Concentration (Scatter Plot)")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred.flatten(), alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual DON Concentration')
    plt.ylabel('Predicted DON Concentration')
    plt.title('Actual vs Predicted DON Concentration')
    plt.grid(True)
    st.pyplot(plt)

    # Plot 5: Training and Validation Loss (Line Plot)
    st.write("### Training and Validation Loss")
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

else:
    st.write("Please upload a CSV file to proceed.")