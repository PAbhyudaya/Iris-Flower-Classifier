from __future__ import annotations
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from iris_app.data import load_iris_df


def build_model(args) -> RandomForestClassifier:
    df, feature_names, _ = load_iris_df()
    X = df[feature_names].values
    y = df["target"].values
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        random_state=42,
    )
    model.fit(X, y)
    return model


essential_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Iris batch predictor")
    p.add_argument("--input", required=True, help="Path to input CSV with sepal/petal columns")
    p.add_argument("--output", required=True, help="Where to write predictions CSV")
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--max-features", default=None)
    p.add_argument("--bootstrap", action="store_true")

    args = p.parse_args(argv)
    model = build_model(args)

    df_in = pd.read_csv(args.input)
    missing = [c for c in essential_cols if c not in df_in.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    probs = model.predict_proba(df_in[essential_cols])
    preds = model.predict(df_in[essential_cols])

    _, _, target_names = load_iris_df()
    result = df_in.copy()
    result["prediction"] = [target_names[i] for i in preds]
    for i, name in enumerate(target_names):
        result[f"prob_{name}"] = probs[:, i]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
