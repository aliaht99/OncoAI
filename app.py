"""
OncoAI — multi-cancer detection demo (brain MRI · lung CT · skin lesions).

Research and education only. Not a medical device.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DEMO = ROOT / "demo_images"
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

st.set_page_config(page_title="OncoAI — Multi-Cancer Detection",
                   page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

# ── task metadata ───────────────────────────────────────────────────────────
TASKS = {
    "Brain MRI": {
        "key": "brain",
        "blurb": "Detects the presence of a tumour on a brain MRI slice.",
        "modality": "MRI",
        "source": "Brain MRI Images for Brain Tumor Detection",
        "malignant": {"tumor"},
    },
    "Lung CT": {
        "key": "lung",
        "blurb": "Classifies a chest CT slice as normal or one of three carcinoma subtypes.",
        "modality": "CT",
        "source": "Chest CT-Scan images (adeno / large cell / squamous / normal)",
        "malignant": {"adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma"},
    },
    "Skin Lesion": {
        "key": "skin",
        "blurb": "Classifies a dermatoscopic image across seven lesion types.",
        "modality": "Dermatoscopy",
        "source": "HAM10000",
        "malignant": {"melanoma", "basal cell carcinoma", "actinic keratoses"},
    },
}

PRETTY = {
    "no_tumor": "No tumour", "tumor": "Tumour",
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large cell carcinoma",
    "squamous.cell.carcinoma": "Squamous cell carcinoma",
    "normal": "Normal",
    "melanoma": "Melanoma",
    "melanocytic_Nevi": "Melanocytic nevus",
    "basal_cell_carcinoma": "Basal cell carcinoma",
    "actinic_keratoses": "Actinic keratoses",
    "benign_keratosis-like_lesions": "Benign keratosis",
    "dermatofibroma": "Dermatofibroma",
    "vascular_lesions": "Vascular lesion",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " ").replace(".", " ").title())


st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#ffffff; }
.hero {
  background:linear-gradient(135deg,#0f766e 0%,#0891b2 55%,#0ea5e9 100%);
  padding:1.6rem 2.2rem;border-radius:18px;margin-bottom:1.1rem;color:#fff;
  box-shadow:0 6px 28px rgba(8,145,178,.32);
}
.hero h1{margin:0;font-size:1.9rem;font-weight:900;color:#fff;letter-spacing:-.5px}
.hero p{margin:.35rem 0 0;font-size:.88rem;color:rgba(255,255,255,.9)}
.pill{display:inline-flex;align-items:center;gap:.4rem;width:100%;justify-content:center;
  white-space:nowrap;padding:.4rem .6rem;border-radius:999px;font-size:.72rem;font-weight:800;
  background:#ecfeff;border:1.5px solid #a5f3fc;color:#0e7490}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.ok{background:#16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.18)}
.off{background:#9ca3af;box-shadow:0 0 0 3px rgba(156,163,175,.18)}
.verdict{padding:1.2rem 1.4rem;border-radius:16px;text-align:center;font-weight:900;
  font-size:1.4rem;margin-bottom:.8rem;box-shadow:0 6px 18px rgba(0,0,0,.14)}
.v-benign{background:linear-gradient(135deg,#d1fae5,#a7f3d0);color:#064e3b;border:2.5px solid #10b981}
.v-watch{background:linear-gradient(135deg,#fef9c3,#fde68a);color:#78350f;border:2.5px solid #f59e0b}
.v-malig{background:linear-gradient(135deg,#fee2e2,#fca5a5);color:#7f1d1d;border:2.5px solid #ef4444}
.disc{background:#fffbeb;border:1.5px solid #fcd34d;border-radius:10px;
  padding:.55rem 1rem;margin:.6rem 0;font-size:.8rem;color:#78350f}
</style>
""", unsafe_allow_html=True)


# ── model loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_task_model(key: str):
    if not TORCH_OK:
        return None, None, None
    path = MODELS / f"{key}_model.pth"
    if not path.exists():
        return None, None, None
    try:
        from common.model import load_checkpoint, get_device
        device = get_device()
        model, ckpt = load_checkpoint(path, device)
        return model, ckpt, device
    except Exception:
        return None, None, None


@st.cache_resource
def load_gate():
    """The modality gate — refuses images that belong to another modality."""
    if not TORCH_OK:
        return None, None, None
    path = MODELS / "modality_gate.pth"
    if not path.exists():
        return None, None, None
    try:
        from common.model import OncoNet, get_device
        device = get_device()
        ck = torch.load(path, map_location=device, weights_only=False)
        gate = OncoNet(len(ck["modalities"]), backbone=ck.get("backbone", "resnet18"),
                       pretrained=False)
        gate.load_state_dict(ck["state_dict"])
        gate.to(device).eval()
        return gate, ck["modalities"], device
    except Exception:
        return None, None, None


@st.cache_data
def load_summary(key: str) -> dict:
    p = RESULTS / key / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    p = MODELS / f"{key}_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(img: Image.Image, size: int, device):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0).to(device)


def gradcam(model, x):
    layer = model.target_layer()
    acts, grads = {}, {}

    def fwd(_m, _i, o):
        acts["v"] = o

    def bwd(_m, _gi, go):
        grads["v"] = go[0]

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)
    try:
        model.zero_grad()
        logits = model(x)
        idx = int(logits.argmax(1))
        logits[0, idx].backward()
        a, g = acts["v"][0], grads["v"][0]
        cam = F.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=x.shape[-2:],
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy()
    except Exception:
        return None
    finally:
        h1.remove()
        h2.remove()


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.asarray(img.convert("RGB").resize((cam.shape[1], cam.shape[0]))) / 255.0
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 OncoAI")
    st.markdown("---")
    task_label = st.selectbox("Cancer type", list(TASKS), index=0)
    task = TASKS[task_label]
    key = task["key"]
    st.caption(task["blurb"])

    st.markdown("---")
    st.markdown("### 📁 Upload an image")
    uploaded = st.file_uploader(f"{task['modality']} image (PNG / JPG)",
                                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    demo_dir = DEMO / key
    demo_files = sorted(demo_dir.glob("*.png")) + sorted(demo_dir.glob("*.jpg"))
    demo_choice = None
    if demo_files:
        st.markdown("### 🧪 …or try a sample")
        names = ["—"] + [p.name for p in demo_files]
        pick = st.selectbox("Bundled example", names, label_visibility="collapsed")
        if pick != "—":
            demo_choice = demo_dir / pick

    st.markdown("---")
    show_cam = st.checkbox("Show Grad-CAM overlay", value=True)
    cam_alpha = st.slider("Overlay intensity", 0.2, 0.8, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 🛡️ Modality gate")
    gate_on = st.checkbox("Refuse mismatched images", value=True,
                          help="Verify the image really is this modality before "
                               "running the diagnostic model.")
    gate_tau = st.slider("Gate confidence floor", 0.5, 0.99, 0.90, 0.01,
                         disabled=not gate_on)
    st.caption("Turn this off to reproduce the silent-failure behaviour "
               "measured in the paper.")

# ── header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 OncoAI — Multi-Cancer Detection</h1>
  <p>Brain MRI · Lung CT · Skin lesions — one interpretable deep-learning platform</p>
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)
for col, (label, meta) in zip(cols, TASKS.items()):
    m, _, _ = load_task_model(meta["key"])
    s = load_summary(meta["key"])
    auc = s.get("macro_auc") or s.get("test_auc")
    with col:
        if m is not None and auc:
            body = f"{label} · AUC {auc:.3f}"
            dot = "ok"
        elif m is not None:
            body, dot = f"{label} · ready", "ok"
        else:
            body, dot = f"{label} · not trained", "off"
        st.markdown(f'<span class="pill"><span class="dot {dot}"></span>{body}</span>',
                    unsafe_allow_html=True)

st.markdown("""
<div class="disc">
  <b>⚕️ Research &amp; education only.</b> OncoAI is <b>not a medical device</b> and is
  <b>not cleared for clinical or diagnostic use</b>. It cannot diagnose cancer and must
  never replace a clinician. Do not upload identifiable patient data.
</div>
""", unsafe_allow_html=True)

tab_predict, tab_perf, tab_about = st.tabs(["🔎 Analyse", "📊 Model performance", "ℹ️ About"])

# ── analyse ─────────────────────────────────────────────────────────────────
with tab_predict:
    model, ckpt, device = load_task_model(key)

    if model is None:
        st.warning(f"No trained model found for **{task_label}**. "
                   f"Train it with `python src/common/train.py {key}`.")
    else:
        src = uploaded or demo_choice
        if src is None:
            st.info("Upload an image in the sidebar, or pick one of the bundled samples, "
                    "to run the model.")
        else:
            img = Image.open(src)
            class_names = ckpt["class_names"]
            size = ckpt["meta"].get("image_size", 224)

            x = preprocess(img, size, device)

            # ── modality gate ────────────────────────────────────────────
            gate, modalities, g_device = load_gate()
            blocked, gate_msg = False, None
            if gate_on and gate is not None and key in modalities:
                with torch.no_grad():
                    gp = torch.softmax(gate(preprocess(img, size, g_device)), 1)[0]
                g_idx = int(gp.argmax())
                g_conf = float(gp[g_idx])
                if g_idx != modalities.index(key):
                    blocked = True
                    gate_msg = (f"This looks like a **{modalities[g_idx]}** image "
                                f"({g_conf:.0%} confident), but you selected "
                                f"**{task_label}**.")
                elif g_conf < gate_tau:
                    blocked = True
                    gate_msg = (f"The gate is only {g_conf:.0%} sure this is a "
                                f"{task['modality']} image, below the "
                                f"{gate_tau:.0%} floor.")

            if blocked:
                st.error(f"🛡️ **Diagnosis withheld.** {gate_msg}")
                st.markdown(
                    "Without this gate the model would still have returned a "
                    "confident answer — that is the failure this project measures. "
                    "Pick the matching cancer type in the sidebar, or upload an "
                    "image of the right modality."
                )
                st.image(img, caption="Rejected input", width="stretch")
                st.stop()
            with torch.no_grad():
                probs = torch.softmax(model(x), 1)[0].cpu().numpy()
            top = int(probs.argmax())
            label = class_names[top]
            conf = float(probs[top])
            is_malignant = label in task["malignant"]

            left, right = st.columns([1, 1], gap="large")

            with left:
                st.image(img, caption=getattr(src, "name", Path(str(src)).name),
                         width="stretch")
                if show_cam:
                    cam = gradcam(model, x)
                    if cam is not None:
                        st.image(overlay(img, cam, cam_alpha), width="stretch",
                                 caption="Grad-CAM — regions driving the prediction")

            with right:
                if is_malignant and conf >= 0.5:
                    css, icon = "v-malig", "⚠️"
                elif is_malignant:
                    css, icon = "v-watch", "⚠️"
                else:
                    css, icon = "v-benign", "✅"
                st.markdown(
                    f'<div class="verdict {css}">{icon} {pretty(label)}<br>'
                    f'<span style="font-size:.9rem;font-weight:500;">'
                    f'confidence {conf:.1%}</span></div>',
                    unsafe_allow_html=True)

                df = (pd.DataFrame({"Class": [pretty(c) for c in class_names],
                                    "Probability": probs})
                      .sort_values("Probability", ascending=False)
                      .reset_index(drop=True))
                st.markdown("**All class probabilities**")
                st.dataframe(
                    df.style.format({"Probability": "{:.1%}"})
                      .bar(subset=["Probability"], color="#67e8f9"),
                    hide_index=True, width="stretch")

                margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])
                if margin < 0.15:
                    st.warning(f"⚠️ Low margin between the top two classes ({margin:.1%}) — "
                               "this prediction is not confident.")

                st.caption("This output is a model score, not a diagnosis. "
                           "Any real concern needs a clinician.")

# ── performance ─────────────────────────────────────────────────────────────
with tab_perf:
    s = load_summary(key)
    if not s:
        st.info("No evaluation summary yet — run "
                f"`python src/common/evaluate.py {key}`.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{s.get('accuracy', s.get('test_accuracy', 0)):.1%}")
        c2.metric("Macro AUC", f"{s.get('macro_auc', s.get('test_auc', 0)):.4f}")
        c3.metric("Classes", len(s.get("classes", [])))

        if s.get("per_class"):
            st.markdown("**Per-class metrics (held-out test set)**")
            t = pd.DataFrame(s["per_class"])
            t["class"] = t["class"].map(pretty)
            st.dataframe(
                t.style.format({c: "{:.3f}" for c in
                                ["sensitivity", "specificity", "precision", "f1", "auc"]}),
                hide_index=True, width="stretch")

        for fname, cap in (("roc_curves.png", "ROC curves (one-vs-rest)"),
                           ("confusion_matrix.png", "Confusion matrix"),
                           ("gradcam.png", "Grad-CAM on test images")):
            p = RESULTS / key / fname
            if p.exists():
                st.markdown(f"**{cap}**")
                st.image(str(p), width="stretch")

# ── about ───────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(f"""
### What this is

OncoAI trains one interpretable image classifier per cancer type and puts them
behind a single interface. Every model is a **transfer-learned CNN**
(ImageNet-pretrained ResNet-18) fine-tuned in two phases, with class-balanced
sampling, early stopping on validation AUC, and **Grad-CAM** so you can see what
the network actually attended to.

| Task | Modality | Classes | Public dataset |
|---|---|---|---|
| Brain | MRI | tumour / no tumour | Brain MRI Images for Brain Tumor Detection |
| Lung | CT | 4 (3 carcinoma subtypes + normal) | Chest CT-Scan images |
| Skin | Dermatoscopy | 7 lesion types | HAM10000 |

### Honest limitations

- These are **slice-level / image-level classifiers**, not detectors. They do not
  localise or stage disease, and Grad-CAM is an explanation aid, not a segmentation.
- The datasets are modest in size and come from specific scanners and populations.
  Accuracy on your own images will very likely be **lower** than the reported
  test numbers — that gap is the norm in medical imaging, not a bug.
- No calibration guarantee: a "90%" score is not a 90% chance of cancer.
- Nothing here has been validated prospectively or reviewed by a regulator.

### Sibling project

The breast-cancer counterpart, **MammoAI**, is a deeper multi-stage system with
clinical-feature fusion, SHAP explanations and calibrated ensembling:
[github.com/aliaht99/MammoAI](https://github.com/aliaht99/MammoAI)
""")
