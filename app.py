"""
Streamlit demo of the cuneiform sign classifier.

- gallery of test examples (pick a class, then an image)
- top-1 prediction + top-3 alternatives with softmax confidence
- warning when the confidence is low (< 50 %)
- Grad-CAM: which part of the image drove the decision

Run:  streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.gradcam import overlay_heatmap
from src.inference import load_checkpoint, predict

# ============== CONFIG ==============
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data/processed")
DEMO_DIR = Path("demo_samples")
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.50

# The trained checkpoint (~123 MB) is not in git. A fresh deployment downloads it from the
# Hugging Face Hub. Pin HF_REVISION to a commit hash of the model repo so a later change to
# that repo cannot alter what the app loads (None = latest).
HF_REPO_ID = "slastrzelec/cuneiform-sign-classifier"
HF_FILENAME = "best_model.pt"
HF_REVISION = None
# =====================================


def _gallery_dir() -> Path:
    """Full test set if present locally, otherwise the small committed demo gallery.
    CUNEIFORM_GALLERY_DIR overrides both (used by the tests)."""
    override = os.environ.get("CUNEIFORM_GALLERY_DIR")
    if override:
        return Path(override)
    test_dir = DATA_DIR / "test"
    return test_dir if test_dir.exists() else DEMO_DIR


def _checkpoint_path() -> Path:
    """Local checkpoint if present, otherwise download from the Hub (cached on disk).
    CUNEIFORM_CHECKPOINT overrides both (used by the tests)."""
    override = os.environ.get("CUNEIFORM_CHECKPOINT")
    if override:
        return Path(override)
    local = CHECKPOINT_DIR / HF_FILENAME
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download  # imported lazily: only needed for the download

    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, revision=HF_REVISION))


@st.cache_resource
def load_model():
    return load_checkpoint(_checkpoint_path())


@st.cache_data
def list_examples(gallery: str):
    """{class name: [image paths]} for the gallery directory."""
    examples = {}
    for class_dir in sorted(Path(gallery).iterdir()):
        if class_dir.is_dir():
            files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
            if files:
                examples[class_dir.name] = files
    return examples


# ================= UI =================
st.set_page_config(page_title="Cuneiform Sign Classifier", layout="wide")
st.title("🏺 Cuneiform sign classifier")
st.caption(
    "Model: ResNet18 (transfer learning), the 30 most frequent signs of the MaiCuBeDa "
    "dataset (Mainz Cuneiform Benchmark Dataset). Pick a class and an example from the "
    "test gallery to see the model's prediction."
)

model, class_names, checkpoint = load_model()
gallery = _gallery_dir()
examples = list_examples(str(gallery))

st.sidebar.header("Pick an example")
selected_class = st.sidebar.selectbox("True class (ground truth)", list(examples.keys()))
selected_file = st.sidebar.selectbox("File", [f.name for f in examples[selected_class]])

image = Image.open(gallery / selected_class / selected_file)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Input image")
    st.image(image, width=250)
    st.caption(f"True class: **{selected_class}**")

top_idx, top_probs, cam = predict(model, image, IMAGE_SIZE)
pred_class = class_names[top_idx[0]]
pred_conf = top_probs[0]

with col2:
    st.subheader("Prediction")
    icon = "✅" if pred_class == selected_class else "❌"
    st.markdown(f"### {icon} {pred_class} ({pred_conf * 100:.1f}%)")

    if pred_conf < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ Low confidence ({pred_conf * 100:.1f}% < "
            f"{LOW_CONFIDENCE_THRESHOLD * 100:.0f}%) — the model is not convinced."
        )

    st.markdown("**Top-3 predictions:**")
    for idx, prob in zip(top_idx, top_probs, strict=True):
        cls = class_names[idx]
        marker = "→" if cls == selected_class else " "
        st.write(f"{marker} {cls}: {prob * 100:.1f}%")
        st.progress(prob)

with col3:
    st.subheader("Grad-CAM")
    st.caption("Red/yellow areas = where the model based its decision")
    st.image(overlay_heatmap(image, cam), width=250)

st.divider()
st.caption(
    f"Model trained for {checkpoint['epoch']} epochs, "
    f"validation macro-F1 = {checkpoint['val_f1']:.3f}. "
    "Data: MaiCuBeDa Hilprecht (CC BY-SA 4.0), MSII rendering."
)
