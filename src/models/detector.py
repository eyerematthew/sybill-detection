import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple, List
import joblib


class SybilDetector:

    def __init__(self, ensemble_method: str = 'voting', use_smote: bool = True):
        self.ensemble_method = ensemble_method
        self.use_smote = use_smote

        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method='hist',
            scale_pos_weight=2.5
        )

        self.lgbm_model = LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.ensemble_model = None
        self.feature_importance = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_smote:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        return X, y

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        X_train, y_train = self.prepare_data(X, y)

        if self.ensemble_method == 'voting':
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('xgb', self.xgb_model),
                    ('lgbm', self.lgbm_model),
                    ('rf', self.rf_model)
                ],
                voting='soft',
                weights=[2, 2, 1],
                n_jobs=-1
            )
            self.ensemble_model.fit(X_train, y_train)

            xgb_importance = self.xgb_model.feature_importances_
            lgbm_importance = self.lgbm_model.feature_importances_
            rf_importance = self.rf_model.feature_importances_

            self.feature_importance = (
                2 * xgb_importance + 2 * lgbm_importance + rf_importance
            ) / 5

        elif self.ensemble_method == 'stacking':
            self.xgb_model.fit(X_train, y_train)
            self.lgbm_model.fit(X_train, y_train)
            self.rf_model.fit(X_train, y_train)

            self.ensemble_model = self.xgb_model
            self.feature_importance = self.xgb_model.feature_importances_

        cv_scores = self._cross_validate(X, y)

        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> List[float]:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        if self.ensemble_method == 'voting':
            model = VotingClassifier(
                estimators=[
                    ('xgb', self.xgb_model),
                    ('lgbm', self.lgbm_model),
                    ('rf', self.rf_model)
                ],
                voting='soft',
                weights=[2, 2, 1]
            )
        else:
            model = self.xgb_model

        scores = cross_val_score(
            model, X, y,
            cv=skf,
            scoring='f1',
            n_jobs=-1
        )

        return scores.tolist()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble_model.predict_proba(X)

    def get_feature_importance(self, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_k)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)


class ThresholdOptimizer:

    @staticmethod
    def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            if metric == 'f1':
                score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            elif metric == 'precision':
                score = precision
            elif metric == 'recall':
                score = recall

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score


def main():
    X = np.load('data/features.npy')
    df = pd.read_csv('data/wallet_dataset.csv')
    y = df['is_sybil'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    detector = SybilDetector(ensemble_method='voting', use_smote=True)
    training_results = detector.train(X_train, y_train)

    print(f"Cross-validation F1 scores: {training_results['cv_scores']}")
    print(f"Mean CV F1: {training_results['mean_cv_score']:.4f} ± {training_results['std_cv_score']:.4f}")

    y_proba = detector.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = ThresholdOptimizer.optimize_threshold(y_test, y_proba)

    print(f"Optimized threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {best_f1:.4f}")

    detector.save('models/sybil_detector.pkl')
    print("Model saved successfully")


if __name__ == "__main__":
    main()
