"""Data utilities for the Iris app.

Only uses standard library, pandas, and scikit-learn.
"""
from __future__ import annotations

from typing import List, Tuple
import pandas as pd
from sklearn import datasets


def load_iris_df() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load the Iris dataset as a pandas DataFrame.

    Returns
    -------
    df : pd.DataFrame
        Columns: sepal_length, sepal_width, petal_length, petal_width, target, target_name
    feature_names : list[str]
        Feature column names
    target_names : list[str]
        Species labels in index order
    """
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=[
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ])
    df["target"] = iris.target
    df["target_name"] = df["target"].map(lambda i: iris.target_names[i])
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = list(iris.target_names)
    return df, feature_names, target_names
