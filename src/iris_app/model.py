"""Model utilities for the Iris app.

Encapsulates training, prediction, and evaluation using scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets


@dataclass
class IrisModel:
    model: RandomForestClassifier
    feature_names: List[str]
    target_names: List[str]

    @classmethod
    def train(
        cls,
        n_estimators: int = 1,
        max_depth: Optional[int] = None,
        max_features: Optional[Union[str, int, float]] = None,
        bootstrap: bool = False,
        random_state: int = 42,
    ) -> Tuple["IrisModel", np.ndarray, np.ndarray]:
        """Train a RandomForestClassifier on the full Iris dataset.

        Returns the model and the full X, y arrays for downstream metrics.
        """
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
        )
        model.fit(X, y)
        return cls(model=model, feature_names=[
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ], target_names=list(iris.target_names)), X, y

    # Predictions
    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(rows)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(rows)

    # Metrics helpers
    @staticmethod
    def split_metrics(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        report = classification_report(y_test, y_pred, target_names=["Setosa", "Versicolor", "Virginica"])
        return train_acc, test_acc, (X_train, X_test, y_train, y_test), cm, report

    @staticmethod
    def cross_val(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, folds: int = 5) -> np.ndarray:
        return cross_val_score(model, X, y, cv=folds)

    @property
    def importances(self) -> np.ndarray:
        return getattr(self.model, "feature_importances_", np.zeros(len(self.feature_names)))
