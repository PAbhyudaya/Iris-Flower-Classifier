from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure src is importable when running `pytest` from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from iris_app.data import load_iris_df
from iris_app.model import IrisModel


def test_training_and_predict_shape():
    model, X, y = IrisModel.train(n_estimators=1, bootstrap=False, random_state=42)
    assert X.shape[0] == y.shape[0] == 150
    # Predict first 5 rows
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 3)


def test_train_accuracy_is_high():
    model, X, y = IrisModel.train(n_estimators=1, bootstrap=False, random_state=42)
    # single tree on training data should fit perfectly
    acc = model.model.score(X, y)
    assert acc >= 0.98
