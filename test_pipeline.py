import unittest
import pandas as pd
import numpy as np
from pipeline import MycotoxinPredictor

class TestMycotoxinPredictor(unittest.TestCase):
    def setUp(self):
        self.file_path = "MLE-Assignment.csv"
        self.predictor = MycotoxinPredictor(self.file_path)

    def test_load_data(self):
        self.predictor.load_data()
        self.assertIsNotNone(self.predictor.data)
        self.assertIsNotNone(self.predictor.df)

    def test_preprocess_data(self):
        self.predictor.load_data()
        self.predictor.preprocess_data()
        self.assertIsNotNone(self.predictor.x_pca)
        self.assertIsNotNone(self.predictor.y_scaled)

    def test_split_data(self):
        self.predictor.load_data()
        self.predictor.preprocess_data()
        self.predictor.split_data()
        self.assertIsNotNone(self.predictor.X_train)
        self.assertIsNotNone(self.predictor.X_test)
        self.assertIsNotNone(self.predictor.y_train)
        self.assertIsNotNone(self.predictor.y_test)

    def test_build_model(self):
        self.predictor.load_data()
        self.predictor.preprocess_data()
        self.predictor.split_data()
        self.predictor.build_model()
        self.assertIsNotNone(self.predictor.model)

if __name__ == "__main__":
    unittest.main()