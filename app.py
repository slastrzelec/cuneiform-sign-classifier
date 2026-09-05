"""
app.py

Interaktywne demo klasyfikatora znakow klinowych (Streamlit).

Funkcje:
- Galeria przykladow z test setu (wybor klasy, potem konkretnego obrazu)
- Predykcja top-1 + top-3 alternatyw z pewnoscia (softmax %)
- Ostrzezenie, gdy pewnosc modelu jest niska (< 50%)
- Grad-CAM: wizualizacja, na jakim fragmencie obrazu model oparl decyzje

Uruchomienie:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from huggingface_hub import hf_hub_download

from src.data import get_transforms, IMAGENET_MEAN, IMAGENET_STD
from train import build_model, CHECKPOINT_DIR

# ============== CONFIG ==============
DATA_DIR = Path("data/processed")
# Full test set (data/processed/test) is gitignored - regenerable locally via
# build_dataset.py. On a fresh deployment (e.g. Streamlit Community Cloud)
# it won't exist, so we fall back to the small curated gallery committed to
# the repo (demo_samples/, ~120 images, a handful per class).
TEST_DIR = DATA_DIR / "test"
DEMO_DIR = Path("demo_samples")
GALLERY_DIR = TEST_DIR if TEST_DIR.exists() else DEMO_DIR
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.50

# Hugging Face Hub repo hosting the trained checkpoint, since best_model.pt
# (~123MB) is gitignored and too large to commit to GitHub directly.
# Update HF_REPO_ID after you've uploaded the model to your own HF account.
HF_REPO_ID = "slastrzelec/cuneiform-sign-classifier"
HF_FILENAME = "best_model.pt"
# =====================================


# ---------- Ladowanie modelu (cache, zeby nie ladowac przy kazdej interakcji) ----------
@st.cache_resource
def load_model():
    local_path = CHECKPOINT_DIR / "best_model.pt"
    if local_path.exists():
        checkpoint_path = local_path
    else:
        # Not present locally (e.g. fresh deployment) - fetch from HF Hub.
        # hf_hub_download caches the file, so this only downloads once.
        checkpoint_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, checkpoint


@st.cache_data
def list_test_examples():
    """Zwraca slownik {charname: [lista sciezek do plikow]}."""
    examples = {}
    for class_dir in sorted(GALLERY_DIR.iterdir()):
        if class_dir.is_dir():
            files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
            if files:
                examples[class_dir.name] = files
    return examples


# ---------- Grad-CAM ----------
class GradCAM:
    """
    Prosta implementacja Grad-CAM dla ResNet - podpina sie pod ostatnia
    warstwe konwolucyjna (layer4), zeby pokazac, ktore rejony obrazu
    najbardziej wplynely na decyzje modelu.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_img: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """Nakłada mapę ciepła Grad-CAM (kolormapa 'jet') na oryginalny obraz."""
    original_resized = original_img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    original_np = np.array(original_resized).astype(float) / 255.0

    heatmap = cm.jet(cam)[:, :, :3]  # usuwamy kanal alpha z colormapy
    overlaid = (1 - alpha) * original_np + alpha * heatmap
    overlaid = np.clip(overlaid, 0, 1)
    return Image.fromarray((overlaid * 255).astype(np.uint8))


def predict(model, image: Image.Image):
    """Zwraca (top3_indices, top3_probs, cam_heatmap)."""
    transform = get_transforms(IMAGE_SIZE, train=False)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)
    input_tensor.requires_grad_(False)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    top3_probs, top3_indices = probs.topk(3)

    # Grad-CAM wymaga osobnego forward+backward z gradientami wlaczonymi
    gradcam = GradCAM(model, model.layer4[-1])
    input_tensor_grad = input_tensor.clone().requires_grad_(True)
    cam = gradcam.generate(input_tensor_grad, top3_indices[0].item())

    return top3_indices.tolist(), top3_probs.tolist(), cam


# ================= UI =================
st.set_page_config(page_title="Cuneiform Sign Classifier", layout="wide")
st.title("🏺 Klasyfikator znaków pisma klinowego")
st.caption(
    "Model: ResNet18 (transfer learning), 30 najczęstszych znaków z datasetu "
    "MaiCuBeDa (Mainz Cuneiform Benchmark Dataset). "
    "Wybierz klasę i przykład z galerii testowej, aby zobaczyć predykcję modelu."
)

model, class_names, checkpoint = load_model()
examples = list_test_examples()

st.sidebar.header("Wybór przykładu")
selected_class = st.sidebar.selectbox("Prawdziwa klasa (ground truth)", list(examples.keys()))
file_options = [f.name for f in examples[selected_class]]
selected_file = st.sidebar.selectbox("Plik", file_options)

selected_path = GALLERY_DIR / selected_class / selected_file
image = Image.open(selected_path)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Obraz wejściowy")
    st.image(image, width=250)
    st.caption(f"Prawdziwa klasa: **{selected_class}**")

top3_idx, top3_probs, cam = predict(model, image)
pred_class = class_names[top3_idx[0]]
pred_conf = top3_probs[0]

with col2:
    st.subheader("Predykcja")
    is_correct = pred_class == selected_class
    icon = "✅" if is_correct else "❌"
    st.markdown(f"### {icon} {pred_class} ({pred_conf*100:.1f}%)")

    if pred_conf < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ Niska pewność predykcji ({pred_conf*100:.1f}% < "
            f"{LOW_CONFIDENCE_THRESHOLD*100:.0f}%) — model nie jest przekonany."
        )

    st.markdown("**Top-3 predykcje:**")
    for idx, prob in zip(top3_idx, top3_probs):
        cls = class_names[idx]
        marker = "→" if cls == selected_class else " "
        st.write(f"{marker} {cls}: {prob*100:.1f}%")
        st.progress(prob)

with col3:
    st.subheader("Grad-CAM")
    st.caption("Czerwone/żółte obszary = miejsca, na których model oparł decyzję")
    overlay = overlay_heatmap(image, cam)
    st.image(overlay, width=250)

st.divider()
st.caption(
    f"Model wytrenowany na epoce {checkpoint['epoch']}, "
    f"val macro-F1={checkpoint['val_f1']:.3f}. "
    f"Dane: MaiCuBeDa Hilprecht (CC BY-SA 4.0), rendering MSII."
)
