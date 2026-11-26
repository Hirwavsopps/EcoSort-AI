import os, sys, json, io, zipfile, time, hashlib, subprocess
from pathlib import Path, PurePosixPath
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(
    os.getenv("ECOSORT_PROJECT_DIR", Path(__file__).resolve().parent.parent)
).resolve()

sys.path.insert(0, str(ROOT))

from src.predictor import EcoSortPredictor

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SPLITS = ROOT / "splits"
INBOX = ROOT / "data_inbox"
META_PATH = MODELS / "model_metadata.json"
METRICS_PATH = REPORTS / "metrics.json"
API_URL = os.getenv("ECOSORT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")
st.title("♻️ EcoSort AI")
st.caption("Waste material classification • model monitoring • controlled retraining")

def read_json(path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default if default is not None else {}

if not META_PATH.exists():
    st.error("Model metadata is missing. Run Notebooks 03 and 04.")
    st.stop()
meta = read_json(META_PATH)
metrics = read_json(METRICS_PATH, {})
classes = meta.get("class_names", [])

# API health and uptime
st.subheader("System status")
health = None
try:
    response = requests.get(f"{API_URL}/health", timeout=3)
    response.raise_for_status()
    health = response.json()
    c1, c2, c3 = st.columns(3)
    c1.metric("API status", "Online")
    c2.metric("Model loaded", "Yes" if health.get("model_loaded") else "No")
    uptime = health.get("uptime_seconds")
    c3.metric("API uptime", f"{uptime/3600:.2f} hours" if uptime is not None else "Not reported")
    st.caption(f"API model version: {health.get('model_version', 'unknown')}")
except Exception:
    st.warning(
        f"API not reachable at {API_URL}. Start it in a separate terminal with "
        "`uvicorn src.app:app --host 127.0.0.1 --port 8000`."
    )

st.subheader("Active model")
m1, m2, m3 = st.columns(3)
m1.metric("Model version", str(meta.get("model_version", "unknown")))
m2.metric("Input size", " × ".join(map(str, meta.get("input_size", [224,224,3])[:2])))
m3.metric("Confidence threshold", f"{float(meta.get('confidence_threshold', 0.70)):.0%}")
st.write("**Recognized material classes:**", ", ".join(classes))

# The metrics file records the initial independent evaluation from Notebook 03.
st.subheader("Evaluation snapshot")
test_metrics = metrics.get("test", {})
if test_metrics:
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy (initial model)", f"{float(test_metrics.get('accuracy', 0)):.1%}")
    b.metric("Macro precision", f"{float(test_metrics.get('precision_macro', test_metrics.get('macro_precision', 0))):.1%}")
    c.metric("Macro recall", f"{float(test_metrics.get('recall_macro', test_metrics.get('macro_recall', 0))):.1%}")
    d.metric("Macro F1", f"{float(test_metrics.get('f1_macro', test_metrics.get('macro_f1', 0))):.1%}")
    st.caption(
        "These are the original Notebook 03 held-out test metrics. If a later model is promoted, "
        "these values are historical and are not a new test evaluation."
    )
else:
    st.info("No reports/metrics.json found. Evaluation metrics will appear after Notebook 03.")

# Visualization 1: class distribution in original training data.
st.subheader("Data and model visualizations")
train_csv = SPLITS / "train.csv"
if train_csv.exists():
    train_df = pd.read_csv(train_csv)
    if "label" in train_df and classes:
        counts = train_df["label"].value_counts().sort_index()
        chart_df = pd.DataFrame({
            "Class": [classes[int(i)] for i in counts.index],
            "Training images": counts.values
        }).set_index("Class")
        st.markdown("**1. Training images per class**")
        st.bar_chart(chart_df)
        st.caption("Interpretation: low-count classes have fewer training examples and may be harder to learn.")
else:
    st.info("Train split CSV not found.")

# Visualization 2: training and validation learning curves.
history = metrics.get("history", {})
if history and all(k in history for k in ["loss", "val_loss", "accuracy", "val_accuracy"]):
    st.markdown("**2. Training and validation curves (initial model)**")
    curve_df = pd.DataFrame({
        "Training loss": history["loss"],
        "Validation loss": history["val_loss"],
        "Training accuracy": history["accuracy"],
        "Validation accuracy": history["val_accuracy"]
    })
    st.line_chart(curve_df)
    st.caption("Interpretation: a widening train/validation gap can indicate overfitting; validation trends guide early stopping.")
else:
    st.info("Training history is not available in reports/metrics.json.")

# Visualization 3: confusion matrix.
cm = metrics.get("confusion_matrix_test")
cm_classes = metrics.get("class_names", classes)
if cm:
    st.markdown("**3. Held-out test confusion matrix (initial model)**")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(len(cm_classes)))
    ax.set_yticks(range(len(cm_classes)))
    ax.set_xticklabels(cm_classes, rotation=60, ha="right")
    ax.set_yticklabels(cm_classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Interpretation: large off-diagonal values show pairs of classes the model confuses.")
else:
    st.info("Confusion matrix not available.")

# Single image prediction
st.subheader("Try one image")
uploaded_image = st.file_uploader(
    "Upload a waste image", type=["jpg", "jpeg", "png", "bmp"], key="single_image"
)
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", width=320)
        if st.button("Classify image", type="primary"):
            with st.spinner("Loading model and classifying..."):
                predictor = EcoSortPredictor(
                    str(MODELS / meta["model_file"]),
                    str(META_PATH),
                    str(ROOT / "disposal_rules.json")
                )
                result = predictor.predict(io.BytesIO(uploaded_image.getvalue()))
            st.write("**Predicted material:**", result["predicted_class"])
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.write("**Top predictions:**", result["top_k"])
            if result["manual_review"]:
                st.warning("Manual review required. No disposal stream is recommended.")
            else:
                st.write("**Configured disposal stream:**", result["disposal_stream"])
            st.caption("Disposal mapping is configurable and must be locally validated before real-world use.")
    except Exception as exc:
        st.error(f"Could not process image: {exc}")

# Bulk labelled upload as ZIP
st.subheader("Add labelled images for future training")
st.write(
    "Upload a ZIP archive with one folder per class. Images are staged; they do not immediately "
    "change the active model."
)
zip_file = st.file_uploader("Labelled dataset ZIP", type=["zip"], key="labelled_zip")
if zip_file is not None and st.button("Validate and stage uploaded images"):
    staged, skipped, errors = 0, 0, []
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_file.getvalue()))
        members = archive.infolist()
        if len(members) > 5000:
            raise ValueError("ZIP contains more than 5,000 entries; split it into smaller uploads.")
        for member in members:
            if member.is_dir():
                continue
            name = member.filename.replace("\\\\", "/")
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                errors.append(f"Rejected unsafe path: {member.filename}")
                continue
            if len(rel.parts) < 2:
                skipped += 1
                continue
            # Accept a wrapper directory, but use the first recognized class folder.
            class_name = next((part for part in rel.parts[:-1] if part in classes), None)
            if class_name is None:
                skipped += 1
                continue
            suffix = Path(rel.name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
                skipped += 1
                continue
            raw = archive.read(member)
            if len(raw) > 20 * 1024 * 1024:
                errors.append(f"Rejected oversized image: {member.filename}")
                continue
            try:
                Image.open(io.BytesIO(raw)).verify()
            except Exception:
                errors.append(f"Rejected unreadable image: {member.filename}")
                continue
            digest = hashlib.sha256(raw).hexdigest()[:16]
            target_dir = INBOX / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{digest}{suffix}"
            if target.exists():
                skipped += 1
                continue
            target.write_bytes(raw)
            staged += 1
        st.success(f"Staged {staged} images. Skipped {skipped}.")
        if errors:
            st.warning("\\n".join(errors[:20]))
    except Exception as exc:
        st.error(f"Could not read ZIP: {exc}")

st.markdown("**Staged image counts**")
staged_counts = {
    c: sum(1 for p in (INBOX / c).glob("*") if p.is_file())
    if (INBOX / c).exists() else 0 for c in classes
}
st.dataframe(pd.DataFrame(
    [{"Class": k, "Staged images": v} for k, v in staged_counts.items()]
), use_container_width=True)

# Retraining
st.subheader("Retrain and validate a candidate")
st.write(
    "The retraining job starts from the active model. It trains with the original training split "
    "plus staged uploads, then compares the candidate with the active model on validation data only."
)
total_staged = sum(staged_counts.values())
if total_staged == 0:
    st.info("Stage labelled images before starting retraining.")
if st.button("Start retraining", type="primary", disabled=(total_staged == 0)):
    with st.spinner("Retraining can take several minutes. Keep this dashboard open..."):
        env = os.environ.copy()
        env["ECOSORT_PROJECT_DIR"] = str(ROOT)
        process = subprocess.run(
            [sys.executable, str(ROOT / "src" / "retrain.py")],
            cwd=str(ROOT), env=env, capture_output=True, text=True
        )
    if process.stdout:
        st.code(process.stdout[-12000:])
    if process.returncode == 0:
        result_path = REPORTS / "last_retraining.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if result.get("promoted"):
            st.success(
                f"Candidate {result.get('candidate_version')} passed validation and was promoted. "
                "Restart the API so it loads the new model."
            )
        else:
            st.warning("Candidate did not pass the promotion gate. The active model was kept.")
        st.json(result)
    else:
        st.error("Retraining failed. The active model was not changed.")
        st.code(process.stderr[-12000:])

last_result = REPORTS / "last_retraining.json"
if last_result.exists():
    st.subheader("Most recent retraining result")
    st.json(read_json(last_result))
