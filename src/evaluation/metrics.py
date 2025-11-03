import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:

    def __init__(self):
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray = None) -> Dict:

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        self.results = metrics
        return metrics

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        print("\n" + "="*60)
        print("SYBIL DETECTION EVALUATION REPORT")
        print("="*60 + "\n")

        report = classification_report(
            y_true, y_pred,
            target_names=['Legitimate', 'Sybil'],
            digits=4
        )
        print(report)

        if self.results:
            print("\nAdditional Metrics:")
            print(f"  Specificity: {self.results['specificity']:.4f}")
            print(f"  False Positive Rate: {self.results['false_positive_rate']:.4f}")
            print(f"  False Negative Rate: {self.results['false_negative_rate']:.4f}")

            if 'roc_auc' in self.results:
                print(f"  ROC AUC: {self.results['roc_auc']:.4f}")
            if 'avg_precision' in self.results:
                print(f"  Average Precision: {self.results['avg_precision']:.4f}")

        print("\n" + "="*60)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Sybil'],
            yticklabels=['Legitimate', 'Sybil']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Sybil Detection')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      save_path: str = None):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Sybil Detection')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    save_path: str = None):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.4f})', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Sybil Detection')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_errors(self, df: pd.DataFrame, y_true: np.ndarray,
                      y_pred: np.ndarray, y_proba: np.ndarray = None) -> pd.DataFrame:

        error_df = df.copy()
        error_df['true_label'] = y_true
        error_df['predicted_label'] = y_pred

        if y_proba is not None:
            error_df['prediction_probability'] = y_proba

        error_df['is_error'] = (y_true != y_pred).astype(int)
        error_df['error_type'] = 'Correct'
        error_df.loc[(y_true == 0) & (y_pred == 1), 'error_type'] = 'False Positive'
        error_df.loc[(y_true == 1) & (y_pred == 0), 'error_type'] = 'False Negative'

        return error_df


class PerformanceAnalyzer:

    @staticmethod
    def analyze_by_transaction_volume(df: pd.DataFrame, y_true: np.ndarray,
                                     y_pred: np.ndarray) -> pd.DataFrame:

        bins = [0, 50, 100, 200, 500, float('inf')]
        labels = ['0-50', '50-100', '100-200', '200-500', '500+']

        analysis_df = pd.DataFrame({
            'transaction_count': df['transaction_count'],
            'true_label': y_true,
            'predicted_label': y_pred
        })

        analysis_df['tx_bin'] = pd.cut(
            analysis_df['transaction_count'],
            bins=bins,
            labels=labels
        )

        results = []
        for bin_label in labels:
            bin_data = analysis_df[analysis_df['tx_bin'] == bin_label]

            if len(bin_data) > 0:
                accuracy = accuracy_score(bin_data['true_label'], bin_data['predicted_label'])
                precision = precision_score(bin_data['true_label'], bin_data['predicted_label'], zero_division=0)
                recall = recall_score(bin_data['true_label'], bin_data['predicted_label'], zero_division=0)
                f1 = f1_score(bin_data['true_label'], bin_data['predicted_label'], zero_division=0)

                results.append({
                    'transaction_range': bin_label,
                    'sample_count': len(bin_data),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        return pd.DataFrame(results)

    @staticmethod
    def analyze_by_account_age(df: pd.DataFrame, y_true: np.ndarray,
                               y_pred: np.ndarray) -> pd.DataFrame:

        analysis_df = pd.DataFrame({
            'account_age_days': (
                pd.Timestamp.now().timestamp() - df['creation_timestamp']
            ) / (24 * 3600),
            'true_label': y_true,
            'predicted_label': y_pred
        })

        bins = [0, 30, 90, 180, 365, float('inf')]
        labels = ['0-30d', '30-90d', '90-180d', '180-365d', '365d+']

        analysis_df['age_bin'] = pd.cut(
            analysis_df['account_age_days'],
            bins=bins,
            labels=labels
        )

        results = []
        for bin_label in labels:
            bin_data = analysis_df[analysis_df['age_bin'] == bin_label]

            if len(bin_data) > 0:
                accuracy = accuracy_score(bin_data['true_label'], bin_data['predicted_label'])
                precision = precision_score(bin_data['true_label'], bin_data['predicted_label'], zero_division=0)
                recall = recall_score(bin_data['true_label'], bin_data['predicted_label'], zero_division=0)
                f1 = f1_score(bin_data['true_label'], bin_data['predicted_label'], zero_division=0)

                results.append({
                    'age_range': bin_label,
                    'sample_count': len(bin_data),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        return pd.DataFrame(results)


def main():
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data/wallet_dataset.csv')
    y = df['is_sybil'].values
    X = np.load('data/features.npy')

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    from src.models.detector import SybilDetector
    detector = SybilDetector.load('models/sybil_detector.pkl')

    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    evaluator.print_report(y_test, y_pred)

    evaluator.plot_confusion_matrix(y_test, y_pred, 'results/confusion_matrix.png')
    evaluator.plot_roc_curve(y_test, y_proba, 'results/roc_curve.png')
    evaluator.plot_precision_recall_curve(y_test, y_proba, 'results/precision_recall_curve.png')

    error_analysis = evaluator.analyze_errors(df_test, y_test, y_pred, y_proba)
    error_analysis.to_csv('results/error_analysis.csv', index=False)

    analyzer = PerformanceAnalyzer()
    tx_analysis = analyzer.analyze_by_transaction_volume(df_test, y_test, y_pred)
    age_analysis = analyzer.analyze_by_account_age(df_test, y_test, y_pred)

    print("\nPerformance by Transaction Volume:")
    print(tx_analysis)

    print("\nPerformance by Account Age:")
    print(age_analysis)


if __name__ == "__main__":
    main()
