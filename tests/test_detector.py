import pytest
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.generator import WalletDataGenerator
from src.features.extractor import FeatureExtractor
from src.models.detector import SybilDetector


class TestWalletDataGenerator:

    def test_generation(self):
        generator = WalletDataGenerator(n_legitimate=100, n_sybil=50, seed=42)
        df, graph = generator.generate()

        assert len(df) == 150
        assert df['is_sybil'].sum() == 50
        assert (df['is_sybil'] == 0).sum() == 100
        assert graph.number_of_nodes() == 150
        assert graph.number_of_edges() > 0

    def test_legitimate_wallets(self):
        generator = WalletDataGenerator(n_legitimate=100, n_sybil=0, seed=42)
        df, _ = generator.generate()

        assert all(df['is_sybil'] == 0)
        assert all(df['transaction_count'] >= 10)
        assert all(df['unique_interactions'] > 0)

    def test_sybil_wallets(self):
        generator = WalletDataGenerator(n_legitimate=0, n_sybil=50, seed=42)
        df, _ = generator.generate()

        assert all(df['is_sybil'] == 1)
        assert 'cluster_id' in df.columns


class TestFeatureExtractor:

    def test_basic_features(self):
        generator = WalletDataGenerator(n_legitimate=50, n_sybil=25, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        features = extractor.extract_basic_features(df)

        assert 'tx_interaction_ratio' in features.columns
        assert 'balance_volatility' in features.columns
        assert len(features) == len(df)

    def test_graph_features(self):
        generator = WalletDataGenerator(n_legitimate=50, n_sybil=25, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        features = extractor.extract_graph_features(df, graph)

        assert 'pagerank' in features.columns
        assert 'clustering_coefficient' in features.columns
        assert all(features['pagerank'] >= 0)

    def test_full_extraction(self):
        generator = WalletDataGenerator(n_legitimate=50, n_sybil=25, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        X = extractor.fit_transform(df, graph)

        assert X.shape[0] == len(df)
        assert X.shape[1] == len(extractor.feature_names)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))


class TestSybilDetector:

    def test_model_training(self):
        generator = WalletDataGenerator(n_legitimate=200, n_sybil=100, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        X = extractor.fit_transform(df, graph)
        y = df['is_sybil'].values

        detector = SybilDetector(ensemble_method='voting', use_smote=False)
        results = detector.train(X, y)

        assert 'cv_scores' in results
        assert results['mean_cv_score'] > 0
        assert len(results['cv_scores']) == 5

    def test_predictions(self):
        generator = WalletDataGenerator(n_legitimate=200, n_sybil=100, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        X = extractor.fit_transform(df, graph)
        y = df['is_sybil'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        detector = SybilDetector(ensemble_method='voting', use_smote=False)
        detector.train(X_train, y_train)

        y_pred = detector.predict(X_test)
        y_proba = detector.predict_proba(X_test)

        assert len(y_pred) == len(X_test)
        assert y_proba.shape == (len(X_test), 2)
        assert all((y_pred == 0) | (y_pred == 1))
        assert all((y_proba >= 0) & (y_proba <= 1).all(axis=1))

    def test_feature_importance(self):
        generator = WalletDataGenerator(n_legitimate=200, n_sybil=100, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        X = extractor.fit_transform(df, graph)
        y = df['is_sybil'].values

        detector = SybilDetector(ensemble_method='voting', use_smote=False)
        detector.train(X, y)

        importance = detector.get_feature_importance(extractor.feature_names)

        assert len(importance) > 0
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestIntegration:

    def test_full_pipeline(self):
        generator = WalletDataGenerator(n_legitimate=300, n_sybil=150, seed=42)
        df, graph = generator.generate()

        extractor = FeatureExtractor()
        X = extractor.fit_transform(df, graph)
        y = df['is_sybil'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        detector = SybilDetector(ensemble_method='voting', use_smote=True)
        detector.train(X_train, y_train)

        y_pred = detector.predict(X_test)
        accuracy = (y_pred == y_test).mean()

        assert accuracy > 0.7
