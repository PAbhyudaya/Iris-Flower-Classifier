from __future__ import annotations
from __future__ import annotations

from typing import List, Tuple
import pandas as pd
from sklearn import datasets


def load_iris_df() -> Tuple[pd.DataFrame, List[str], List[str]]:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=[
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ])
    df["target"] = iris.target
    df["target_name"] = df["target"].map(lambda i: iris.target_names[i])
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = list(iris.target_names)
    return df, feature_names, target_names
