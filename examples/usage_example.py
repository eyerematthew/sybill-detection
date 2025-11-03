import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import WalletDataGenerator
from src.features.extractor import FeatureExtractor
from src.models.detector import SybilDetector
from src.evaluation.metrics import ModelEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd


def example_training():
    print("="*60)
    print("EXAMPLE: Training Sybil Detection Model")
    print("="*60)

    print("\n1. Generating dataset...")
    generator = WalletDataGenerator(n_legitimate=1000, n_sybil=400, seed=42)
    df, graph = generator.generate()
    print(f"   Generated {len(df)} wallets")

    print("\n2. Extracting features...")
    extractor = FeatureExtractor()
    X = extractor.fit_transform(df, graph)
    y = df['is_sybil'].values
    print(f"   Extracted {X.shape[1]} features")

    print("\n3. Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    detector = SybilDetector(ensemble_method='voting', use_smote=True)
    results = detector.train(X_train, y_train)
    print(f"   CV F1 Score: {results['mean_cv_score']:.4f}")

    print("\n4. Evaluating...")
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)

    print(f"\n   Results:")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall: {metrics['recall']:.4f}")
    print(f"   - F1 Score: {metrics['f1']:.4f}")
    print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")

    return detector, extractor, df, graph


def example_prediction(detector, extractor, df, graph):
    print("\n" + "="*60)
    print("EXAMPLE: Making Predictions")
    print("="*60)

    sample_wallets = df.sample(5, random_state=42)

    print("\nPredicting 5 random wallets:")
    print("-" * 60)

    for idx, row in sample_wallets.iterrows():
        sample_df = pd.DataFrame([row])
        sample_graph = graph.subgraph([row['address']])

        X_sample = extractor.transform(sample_df, graph)
        y_proba = detector.predict_proba(X_sample)[0]

        print(f"\nWallet: {row['address'][:10]}...")
        print(f"  True Label: {'Sybil' if row['is_sybil'] == 1 else 'Legitimate'}")
        print(f"  Predicted: {'Sybil' if y_proba[1] > 0.75 else 'Legitimate'}")
        print(f"  Confidence: {y_proba[1]:.4f}")
        print(f"  Risk Score: {y_proba[1] * 100:.2f}/100")


def example_feature_importance(detector, extractor):
    print("\n" + "="*60)
    print("EXAMPLE: Feature Importance Analysis")
    print("="*60)

    importance = detector.get_feature_importance(extractor.feature_names, top_k=10)

    print("\nTop 10 Most Important Features:")
    print("-" * 60)
    for idx, row in importance.iterrows():
        print(f"{idx+1:2d}. {row['feature']:30s} {row['importance']:.4f}")


def main():
    print("\n" + "="*60)
    print("SYBIL WALLET DETECTION - USAGE EXAMPLES")
    print("="*60)

    detector, extractor, df, graph = example_training()

    example_prediction(detector, extractor, df, graph)

    example_feature_importance(detector, extractor)

    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
