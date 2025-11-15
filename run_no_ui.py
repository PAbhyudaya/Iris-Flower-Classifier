from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from iris_app.data import load_iris_df
from iris_app.model import IrisModel


def print_confusion_matrix(cm, target_names):
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("        " + "  ".join([f"{name[:8]:>8}" for name in target_names]))
    for i, row in enumerate(cm):
        print(f"{target_names[i][:8]:>8} " + "  ".join([f"{val:8d}" for val in row]))


def main():
    df, feature_names, target_names = load_iris_df()
    X = df[feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=42,
        ),
        X,
        y,
        cv=rskf,
        n_jobs=None,
    )

    # Evaluate on the entire dataset
    y_pred_full = model.predict(X)
    cm_full = confusion_matrix(y, y_pred_full, labels=[0, 1, 2])
    report_full = classification_report(
        y,
        y_pred_full,
        target_names=[n.title() for n in target_names],
        digits=3,
        zero_division=0,
    )
    balanced_acc_full = balanced_accuracy_score(y, y_pred_full)

    print("=== Iris Model (no UI) ===")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy : {test_acc:.3f}")
    print(f"Balanced acc  : {balanced_acc_full:.3f}")
    print()
    print_confusion_matrix(cm_full, [n.title() for n in target_names])
    print()
    print("Classification report:")
    print(report_full)


if __name__ == "__main__":
    main()