import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

from data.generator import WalletDataGenerator
from features.extractor import FeatureExtractor
from models.detector import SybilDetector, ThresholdOptimizer
from evaluation.metrics import ModelEvaluator, PerformanceAnalyzer


class SybilDetectionPipeline:

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.generator = None
        self.extractor = None
        self.detector = None
        self.evaluator = None

        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    @staticmethod
    def _default_config():
        return {
            'n_legitimate': 5000,
            'n_sybil': 2000,
            'test_size': 0.2,
            'random_state': 42,
            'ensemble_method': 'voting',
            'use_smote': True,
            'optimize_threshold': True
        }

    def generate_dataset(self):
        print("="*60)
        print("STEP 1: Generating synthetic dataset")
        print("="*60)

        self.generator = WalletDataGenerator(
            n_legitimate=self.config['n_legitimate'],
            n_sybil=self.config['n_sybil'],
            seed=self.config['random_state']
        )

        wallets_df, graph = self.generator.generate()

        wallets_df.to_csv('data/wallet_dataset.csv', index=False)
        nx.write_gpickle(graph, 'data/transaction_graph.gpickle')

        print(f"✓ Generated {len(wallets_df)} wallets")
        print(f"  - Legitimate: {(wallets_df['is_sybil'] == 0).sum()}")
        print(f"  - Sybil: {(wallets_df['is_sybil'] == 1).sum()}")
        print(f"  - Graph edges: {graph.number_of_edges()}")
        print(f"✓ Dataset saved to data/")

        return wallets_df, graph

    def extract_features(self, wallets_df: pd.DataFrame, graph: nx.DiGraph):
        print("\n" + "="*60)
        print("STEP 2: Extracting features")
        print("="*60)

        self.extractor = FeatureExtractor()
        X = self.extractor.fit_transform(wallets_df, graph)

        np.save('data/features.npy', X)
        joblib.dump(self.extractor, 'models/feature_extractor.pkl')

        print(f"✓ Extracted {X.shape[1]} features from {X.shape[0]} wallets")
        print(f"✓ Feature extractor saved to models/")

        return X

    def train_model(self, X: np.ndarray, y: np.ndarray):
        print("\n" + "="*60)
        print("STEP 3: Training detection model")
        print("="*60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        self.detector = SybilDetector(
            ensemble_method=self.config['ensemble_method'],
            use_smote=self.config['use_smote']
        )

        print("Training ensemble model...")
        training_results = self.detector.train(X_train, y_train)

        print(f"✓ Model training completed")
        print(f"  - Cross-validation F1: {training_results['mean_cv_score']:.4f} ± {training_results['std_cv_score']:.4f}")

        if self.config['optimize_threshold']:
            y_proba = self.detector.predict_proba(X_test)[:, 1]
            threshold, score = ThresholdOptimizer.optimize_threshold(y_test, y_proba)
            print(f"  - Optimized threshold: {threshold:.3f}")
            print(f"  - Optimized F1: {score:.4f}")

        self.detector.save('models/sybil_detector.pkl')
        print(f"✓ Model saved to models/")

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                      df_test: pd.DataFrame):
        print("\n" + "="*60)
        print("STEP 4: Evaluating model")
        print("="*60)

        y_pred = self.detector.predict(X_test)
        y_proba = self.detector.predict_proba(X_test)[:, 1]

        self.evaluator = ModelEvaluator()
        metrics = self.evaluator.evaluate(y_test, y_pred, y_proba)

        self.evaluator.print_report(y_test, y_pred)

        self.evaluator.plot_confusion_matrix(y_test, y_pred, 'results/confusion_matrix.png')
        self.evaluator.plot_roc_curve(y_test, y_proba, 'results/roc_curve.png')
        self.evaluator.plot_precision_recall_curve(y_test, y_proba, 'results/precision_recall_curve.png')

        print(f"\n✓ Evaluation plots saved to results/")

        error_analysis = self.evaluator.analyze_errors(df_test, y_test, y_pred, y_proba)
        error_analysis.to_csv('results/error_analysis.csv', index=False)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('results/metrics.csv', index=False)

        print(f"✓ Evaluation results saved to results/")

        return metrics

    def analyze_performance(self, df_test: pd.DataFrame, y_test: np.ndarray,
                          y_pred: np.ndarray):
        print("\n" + "="*60)
        print("STEP 5: Performance analysis")
        print("="*60)

        analyzer = PerformanceAnalyzer()

        tx_analysis = analyzer.analyze_by_transaction_volume(df_test, y_test, y_pred)
        age_analysis = analyzer.analyze_by_account_age(df_test, y_test, y_pred)

        print("\nPerformance by Transaction Volume:")
        print(tx_analysis.to_string(index=False))

        print("\nPerformance by Account Age:")
        print(age_analysis.to_string(index=False))

        tx_analysis.to_csv('results/performance_by_tx_volume.csv', index=False)
        age_analysis.to_csv('results/performance_by_account_age.csv', index=False)

        print(f"\n✓ Performance analysis saved to results/")

    def analyze_feature_importance(self):
        print("\n" + "="*60)
        print("STEP 6: Feature importance analysis")
        print("="*60)

        importance_df = self.detector.get_feature_importance(
            self.extractor.feature_names,
            top_k=20
        )

        print("\nTop 20 Most Important Features:")
        print(importance_df.to_string(index=False))

        importance_df.to_csv('results/feature_importance.csv', index=False)
        print(f"\n✓ Feature importance saved to results/")

    def run_full_pipeline(self):
        start_time = datetime.now()

        print("\n" + "="*70)
        print("SYBIL WALLET DETECTION PIPELINE")
        print("="*70)

        wallets_df, graph = self.generate_dataset()

        y = wallets_df['is_sybil'].values
        X = self.extract_features(wallets_df, graph)

        X_train, X_test, y_train, y_test = self.train_model(X, y)

        df_test = wallets_df.iloc[
            train_test_split(
                range(len(wallets_df)),
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )[1]
        ]

        metrics = self.evaluate_model(X_test, y_test, df_test)

        y_pred = self.detector.predict(X_test)
        self.analyze_performance(df_test, y_test, y_pred)

        self.analyze_feature_importance()

        elapsed_time = (datetime.now() - start_time).total_seconds()

        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70)
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Final F1 Score: {metrics['f1']:.4f}")
        print(f"Final Precision: {metrics['precision']:.4f}")
        print(f"Final Recall: {metrics['recall']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("="*70 + "\n")


def main():
    pipeline = SybilDetectionPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
