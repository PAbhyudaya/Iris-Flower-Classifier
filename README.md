
# Iris Flower Classifier - AI in Action 🌸

An interactive Streamlit ML demo using the classic Iris dataset. This advanced version adds model tuning, metrics, a confusion matrix, feature importances, flexible plots, and batch predictions.

## Features
- Loads the Iris dataset from `sklearn.datasets`
- Sidebar hyperparameters for Random Forest (trees, max depth, max features, bootstrap)
- Metrics: Train accuracy, Test accuracy, and 5-fold Cross-Validation
- Confusion matrix heatmap (on test split) + classification report
- Feature importances bar chart
- Tabs for a clean UX:
  - 🔮 Predict — sliders + instant prediction and class confidences
  - 📚 Dataset — explore scatter plots with selectable axes
  - 🧪 Model & Metrics — detailed metrics, confusion matrix, importances
  - 📦 Batch Predict — upload CSV, get predictions and download results
- Big, bold result box by species color:
  - Green = Setosa, Blue = Versicolor, Purple = Virginica
- Footer: “Built with Streamlit & scikit-learn”

## Project structure
- `app.py` — Streamlit application (imports from `src/iris_app`)
- `src/iris_app/` — Modular package
  - `data.py` — dataset loading
  - `model.py` — training, evaluation, cross-validation helpers
  - `viz.py` — Plotly figure builders (scatter, confusion matrix, importances)
  - `__init__.py` — package metadata
- `cli.py` — Command-line batch predictions
- `tests/` — Minimal unit tests (optional for devs)
- `requirements.txt` — Runtime dependencies
- `requirements-dev.txt` — Optional dev tools (pytest, black, flake8)

## Prerequisites
- Python 3.9+ recommended
- Windows PowerShell or your preferred terminal

## Installation (Windows PowerShell)
```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app
```powershell
streamlit run app.py
```
Then open the local URL shown in the terminal (usually http://localhost:8501).

## How to use
1. Use the sidebar to adjust Random Forest settings (e.g., n_estimators, max_depth). Metrics will update automatically (Train/Test/CV).
2. In the 🔮 Predict tab, move the four sliders; see the prediction and class confidences update instantly.
3. In the 📚 Dataset tab, pick any two features for the scatter plot to explore clusters.
4. In the 🧪 Model & Metrics tab, inspect the confusion matrix, feature importances, and the classification report.
5. In the 📦 Batch Predict tab, upload a CSV to get predictions for many rows and download a results file.

## CSV format for batch prediction
Required columns (case-sensitive):
```
sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
6.2,2.8,4.8,1.8
```

## Command-line batch predictions (optional)
```powershell
python cli.py --input data.csv --output predictions.csv --n-estimators 100 --max-depth 5 --max-features sqrt --bootstrap
```

## Run tests (optional)
```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Sample output (Predict tab)
Example with typical Setosa-like values:
- Sepal Length: 5.1
- Sepal Width: 3.5
- Petal Length: 1.5
- Petal Width: 0.2

Expected result:
- Prediction: Setosa (green box)
- Confidence: Setosa ≈ 100%, Versicolor ≈ 0%, Virginica ≈ 0%

## License
This example is provided for learning and demonstration purposes.

# Iris-Flower-Classifier
An upgraded Streamlit web app for the classic Iris dataset
