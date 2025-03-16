import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MycotoxinPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.pca = PCA(n_components=20)

    def load_data(self):
        logger.info("Loading data...")
        self.data = pd.read_csv(self.file_path)
        self.df = self.data.copy()
        logger.info("Data loaded successfully.")

    def preprocess_data(self):
        logger.info("Preprocessing data...")
        self.df.dropna(inplace=True)
        X = self.df.iloc[:, 1:-1]
        Y = self.df.iloc[:, -1]
        x_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(Y.values.reshape(-1, 1)).ravel()
        self.x_pca = self.pca.fit_transform(x_scaled)
        self.y_scaled = y_scaled
        logger.info("Data preprocessing completed.")

    def split_data(self):
        logger.info("Splitting data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x_pca, self.y_scaled, test_size=0.2, random_state=42
        )
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        logger.info("Data splitting completed.")

    def build_model(self):
        logger.info("Building CNN model...")
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.X_train.shape[1], 1)),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info("CNN model built and compiled.")

    def train_model(self):
        logger.info("Training model...")
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=100, batch_size=32, callbacks=[es]
        )
        logger.info("Model training completed.")

    def evaluate_model(self):
        logger.info("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        logger.info(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')

    def plot_results(self):
        logger.info("Plotting results...")
        y_pred = self.model.predict(self.X_test)
        plt.figure(figsize=(6, 6))
        sns.regplot(x=self.y_test, y=y_pred.flatten(), scatter_kws={'s': 10}, line_kws={"color": "red"})
        plt.xlabel('Actual DON Concentration')
        plt.ylabel('Predicted DON Concentration')
        plt.title('Actual vs Predicted DON Concentration (with Regression Line)')
        plt.grid(True)
        plt.show()
        logger.info("Results plotted.")

def main(file_path):
    predictor = MycotoxinPredictor(file_path)
    predictor.load_data()
    predictor.preprocess_data()
    predictor.split_data()
    predictor.build_model()
    predictor.train_model()
    predictor.evaluate_model()
    predictor.plot_results()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        logger.error("Usage: python pipeline.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])