"""
Iris Flower Classifier - AI in Action

An upgraded Streamlit web app for the classic Iris dataset with:
- Sidebar hyperparameters for the Random Forest (n_estimators, max_depth, max_features, bootstrap)
- Metrics: Train Accuracy, Test Accuracy, and 5-fold Cross-Validation score
- Tabs: Predict, Dataset, Model & Metrics, Batch Predict
- Confusion matrix heatmap and feature importances chart
- Flexible scatter plot axes
- Batch CSV predictions with downloadable results

Libraries used: streamlit, scikit-learn, pandas, plotly
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px

# Allow importing local package from ./src when running via `streamlit run app.py`
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.append(str(SRC))

from iris_app.data import load_iris_df
from iris_app.viz import (
	PREDICTION_COLORS,
	scatter_species,
)

# ---------------------- Image assets ----------------------
# Expecting images under ./img with filenames matching these mappings.
IMG_DIR = ROOT / "img"
SPECIES_IMAGE_MAP = {
	"setosa": IMG_DIR / "Iris_Setosa.jpeg",
	"versicolor": IMG_DIR / "Iris_Versicolor.jpg",
	"virginica": IMG_DIR / "iris_virginica.jpg",
}
COMPOSITE_IMAGE = IMG_DIR / "Irish_flowers.png"

# ---------------------- Page config & Styling ----------------------
st.set_page_config(
	page_title="Iris Flower Classifier - AI in Action",
	layout="centered",
	page_icon="🌸",
)

# Small CSS to make the prediction box stand out and center content

BOX_CSS = """
<style>
.prediction-box {
  padding: 1.25rem 1rem;
  border-radius: 0.75rem;
  color: white;
  font-weight: 800;
  font-size: 1.25rem;
  text-align: center;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.footer-text { text-align: center; color: #6b7280; font-size: 0.9rem; }
.small-note { color: #6b7280; font-size: 0.85rem; }
</style>
"""
st.markdown(BOX_CSS, unsafe_allow_html=True)

# ---------------------- Data & Model ----------------------
@st.cache_data(show_spinner=False)
def load_data():
	"""Load the Iris dataset and return a DataFrame and metadata.

	Returns
	-------
	df : pd.DataFrame
		Data with columns: sepal_length, sepal_width, petal_length, petal_width, target, target_name
	feature_names : list[str]
		Names of the numeric feature columns
	target_names : list[str]
		Species names in index order
	iris_raw : sklearn.utils.Bunch
		Raw iris dataset object for reference
	"""
	df, feature_names, target_names = load_iris_df()
	return df, feature_names, target_names, None


@st.cache_resource(show_spinner=False)
def train_model(n_estimators: int, max_depth: int | None, max_features: str | int | float | None,
				bootstrap: bool, random_state: int):
	"""Train a Random Forest on the full dataset with user-controlled hyperparameters.

	Notes
	-----
	- For perfect training accuracy (demo of overfitting), set n_estimators=1 and bootstrap=False.

	Returns
	-------
	model : RandomForestClassifier
	X : np.ndarray
	y : np.ndarray
	"""
	# Use the same dataset loaded for the UI
	df, feature_names, _ = load_iris_df()
	X = df[feature_names].values
	y = df["target"].values

	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		max_features=max_features,
		bootstrap=bootstrap,
		random_state=random_state,
	)
	model.fit(X, y)
	return model, X, y


# ---------------------- App UI ----------------------
st.title("Iris Flower Classifier - AI in Action 🌸🤖")
st.write("Explore how a tiny ML model classifies iris flowers from sepal and petal measurements.")

# Load data
df, FEATURE_NAMES, TARGET_NAMES, iris_raw = load_data()


# Model hyperparameters (fixed, not interactive)
n_estimators = 100
max_depth = None
max_features = None
bootstrap = False
test_size = 0.2

# Train with current settings
model, X_all, y_all = train_model(
	n_estimators=n_estimators,
	max_depth=max_depth,
	max_features=max_features,
	bootstrap=bootstrap,
	random_state=42,
)

# Metrics: train/test split and CV
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42, stratify=y_all)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
cv_model = RandomForestClassifier(
	n_estimators=n_estimators,
	max_depth=max_depth,
	max_features=max_features,
	bootstrap=bootstrap,
	random_state=42,
)
cv_scores = cross_val_score(cv_model, X_all, y_all, cv=5)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Train Accuracy", f"{train_acc*100:.0f}%")
col_b.metric("Test Accuracy", f"{test_acc*100:.0f}%")
col_c.metric("5-fold CV", f"{cv_scores.mean()*100:.0f}%", help=f"Std: {cv_scores.std()*100:.1f}%")

st.divider()

# Tabs
tab_predict, tab_dataset, tab_model, tab_batch = st.tabs(["🔮 Predict", "📚 Dataset", "🧪 Model & Metrics", "📦 Batch Predict"])

# ---- Predict Tab ----

with tab_predict:
	st.subheader("Try it yourself ✋")
	st.caption("Drag the sliders and see the prediction update instantly.")

	# Show species image above sliders, bigger size
	input_row = None
	pred_key = None
	pred_name = None
	img_path = None
	# Use default slider values for initial image
	default_values = dict(sepal_length=5.1, sepal_width=3.5, petal_length=1.5, petal_width=0.2)
	col_left, col_right = st.columns(2)
	with col_left:
		sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, default_values['sepal_length'], 0.1)
		petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, default_values['petal_length'], 0.1)
	with col_right:
		sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, default_values['sepal_width'], 0.1)
		petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, default_values['petal_width'], 0.1)

	input_row = [[sepal_length, sepal_width, petal_length, petal_width]]
	proba = model.predict_proba(input_row)[0]
	pred_idx = int(np.argmax(proba))
	pred_name = TARGET_NAMES[pred_idx]
	pred_key = pred_name.lower()
	img_path = SPECIES_IMAGE_MAP.get(pred_key)
	if img_path and img_path.exists():
		st.image(str(img_path), caption=f"{pred_name.title()} sample", width=700, output_format="JPEG")
		st.markdown("<style>img {height: 420px !important; object-fit: cover;}</style>", unsafe_allow_html=True)
	else:
		st.caption("Species image not found in ./img")

	color = PREDICTION_COLORS.get(pred_key, "#0ea5e9")
	st.markdown(
		f'<div class="prediction-box" style="background:{color}">'
		f'🌺 Prediction: {pred_name.title()}</div>',
		unsafe_allow_html=True,
	)

	st.write("")
	st.subheader("Confidence by class 📊")
	for i, name in enumerate(TARGET_NAMES):
		pct = int(round(proba[i] * 100))
		st.write(f"{name.title()} — {pct}%")
		st.progress(pct, text=None)

# ---- Dataset Tab ----
with tab_dataset:
	st.subheader("Iris dataset overview 🌈")
	# Composite overview image
	if COMPOSITE_IMAGE.exists():
		import base64
		from pathlib import Path
		img_bytes = Path(COMPOSITE_IMAGE).read_bytes()
		img_b64 = base64.b64encode(img_bytes).decode()
		img_html = f"""
		<div style='overflow-x:auto; width:100%;'>
		  <img src='data:image/png;base64,{img_b64}' style='width:100%; height:auto; display:block;' alt='All three Iris species'>
		  <div style='font-size:0.9em; color:gray;'>All three Iris species</div>
		</div>
		"""
		st.markdown(img_html, unsafe_allow_html=True)
	else:
		st.caption("Composite image not found in ./img")
	st.dataframe(df, use_container_width=True)

	x_axis = st.selectbox("X axis", FEATURE_NAMES, index=2)
	y_axis = st.selectbox("Y axis", FEATURE_NAMES, index=3)

	fig = scatter_species(df, x_axis, y_axis)
	st.plotly_chart(fig, use_container_width=True)

# ---- Model & Metrics Tab ----
with tab_model:
	st.subheader("Metrics & Insights 📈")

	# Confusion Matrix (test split)
	y_pred_test = model.predict(X_test)
	cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
	cm_df = pd.DataFrame(cm, index=[n.title() for n in TARGET_NAMES], columns=[n.title() for n in TARGET_NAMES])
	cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
					   labels=dict(x="Predicted", y="Actual", color="Count"),
					   title="Confusion Matrix (Test Set)")
	cm_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
	st.plotly_chart(cm_fig, use_container_width=True)

	# Feature Importances
	st.subheader("Feature importances 🌟")
	importances = getattr(model, "feature_importances_", np.zeros(len(FEATURE_NAMES)))
	imp_df = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances}).sort_values("importance", ascending=True)
	imp_fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
					 title="Which features drive the model?",
					 labels={"importance": "Importance", "feature": "Feature"},
					 color="feature", color_discrete_sequence=px.colors.qualitative.Set2)
	st.plotly_chart(imp_fig, use_container_width=True)

	# Classification report
	st.subheader("Classification report 🧾")
	report = classification_report(y_test, y_pred_test, target_names=[n.title() for n in TARGET_NAMES])
	st.code(report, language="text")

# ---- Batch Predict Tab ----
with tab_batch:
	st.subheader("Upload a CSV for batch predictions 📦")
	st.caption("CSV must contain columns: sepal_length, sepal_width, petal_length, petal_width")
	uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

	if uploaded is not None:
		try:
			df_in = pd.read_csv(uploaded)
		except Exception as e:
			st.error(f"Could not read CSV: {e}")
			df_in = None

		if df_in is not None:
			missing = [c for c in FEATURE_NAMES if c not in df_in.columns]
			if missing:
				st.error(f"Missing required columns: {missing}")
			else:
				probs = model.predict_proba(df_in[FEATURE_NAMES])
				preds = model.predict(df_in[FEATURE_NAMES])
				result = df_in.copy()
				result["prediction"] = [TARGET_NAMES[i] for i in preds]
				for i, name in enumerate(TARGET_NAMES):
					result[f"prob_{name}"] = probs[:, i]

				st.success("Predictions ready!")
				st.dataframe(result.head(20), use_container_width=True)

				csv_bytes = result.to_csv(index=False).encode("utf-8")
				st.download_button("Download predictions as CSV", data=csv_bytes, file_name="iris_predictions.csv", mime="text/csv")

st.write("\n")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer-text">Built with <b>Streamlit</b> & <b>scikit-learn</b> • 💡 Learn by doing</div>', unsafe_allow_html=True)
