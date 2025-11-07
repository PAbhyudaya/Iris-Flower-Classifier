"""Visualization utilities for the Iris app using Plotly."""
from __future__ import annotations

from typing import Dict, List
import pandas as pd
import numpy as np
import plotly.express as px


PREDICTION_COLORS: Dict[str, str] = {
    "setosa": "#22c55e",
    "versicolor": "#3b82f6",
    "virginica": "#a855f7",
}


def scatter_species(df: pd.DataFrame, x: str, y: str) -> "px.Figure":
    color_map = {
        "setosa": PREDICTION_COLORS["setosa"],
        "versicolor": PREDICTION_COLORS["versicolor"],
        "virginica": PREDICTION_COLORS["virginica"],
    }
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="target_name",
        color_discrete_map=color_map,
        hover_data=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        labels={x: f"{x.replace('_',' ').title()} (cm)", y: f"{y.replace('_',' ').title()} (cm)", "target_name": "Species"},
        title=f"{x.replace('_',' ').title()} vs {y.replace('_',' ').title()} by species",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def confusion_matrix_figure(cm: np.ndarray, target_names: List[str]) -> "px.Figure":
    df_cm = pd.DataFrame(cm, index=[n.title() for n in target_names], columns=[n.title() for n in target_names])
    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix (Test Set)")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def feature_importances_figure(features: List[str], importances: np.ndarray) -> "px.Figure":
    df_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=True)
    fig = px.bar(df_imp, x="importance", y="feature", orientation="h",
                 title="Which features drive the model?",
                 labels={"importance": "Importance", "feature": "Feature"},
                 color="feature", color_discrete_sequence=px.colors.qualitative.Set2)
    return fig
