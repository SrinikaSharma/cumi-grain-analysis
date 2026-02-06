# ============================================================
# IMPORTANT: MUST BE FIRST
# ============================================================
import os
os.environ["TORCHVISION_DISABLE_NMS"] = "1"

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ============================================================
# SESSION STATE INIT
# ============================================================
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CV Grain Analysis",
    page_icon="🔬",
    layout="centered"
)

# ============================================================
# CSS FIXES
# ============================================================
st.markdown(
    """
    <style>
    div[data-testid="stFileUploader"] ul { display: none; }
    div[data-testid="stFileUploader"] > div > div:last-child { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <h1 style="text-align:center;">🔬 CV Software for Coated Grain Analysis</h1>
    <p style="text-align:center; color:gray;">
    Automated color consistency & coating quality estimation
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# STEP 0 — GRAIN TYPE SELECTION
# ============================================================
st.markdown("### Step 0: Select Grain Type")

grain_type = st.radio(
    "",
    ["🔴 Red grains", "⚫ Black grains", "🟣 Purple grains", "⚪ White grains"],
    horizontal=True
)

# ============================================================
# CHECKPOINT + SD_MAX
# ============================================================
CHECKPOINT_MAP = {
    "🔴 Red grains": {
        "ckpt": "checkpoints/resnet50_6d_C_dualhead_onlyDiffusion.pth",
        "sd_max": 4.74
    },
    "⚫ Black grains": {
        "ckpt": "checkpoints/resnet50_6d_C_dualhead_onlyDiffusion_BLACKGRAINS.pth",
        "sd_max": 6.428001245
    },
    "🟣 Purple grains": {
        "ckpt": "checkpoints/resnet50_6d_C_dualhead_onlyDiffusion_PURPLEGRAINS.pth",
        "sd_max": 13.02
    },
    "⚪ White grains": {
        "ckpt": "checkpoints/resnet50_6d_C_dualhead_onlyDiffusion_WHITEGRAINS.pth",
        "sd_max": 18.24
    }
}

CHECKPOINT_PATH = CHECKPOINT_MAP[grain_type]["ckpt"]
sd_max = CHECKPOINT_MAP[grain_type]["sd_max"]

st.success(f"✔ Loaded model for **{grain_type.split(' ',1)[1]}**")
st.divider()

# ============================================================
# IMAGE GRID FUNCTION
# ============================================================
def make_image_grid_pil(images):
    grid = Image.new("RGB", (900, 600))
    for i, img in enumerate(images):
        img = img.convert("RGB").resize((300, 300), Image.BILINEAR)
        grid.paste(img, ((i % 3) * 300, (i // 3) * 300))
    return grid

# ============================================================
# MODEL
# ============================================================
class ResNetDualHead(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.rgb_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.sd_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.cat([self.rgb_head(x), self.sd_head(x)], dim=1)

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetDualHead().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

model, device = load_model(CHECKPOINT_PATH)

# ============================================================
# TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# STEP 1 — UPLOAD WITH REMOVE OPTION
# ============================================================
st.markdown("### 📤 Step 1: Upload Exactly 6 Images")

remaining = 6 - len(st.session_state.uploaded_images)

if remaining > 0:
    new_files = st.file_uploader(
        f"Upload up to {remaining} image(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{len(st.session_state.uploaded_images)}"
    )

    if new_files:
        for f in new_files:
            if len(st.session_state.uploaded_images) < 6:
                st.session_state.uploaded_images.append(f)
            else:
                st.warning("⚠️ You cannot upload more than 6 images.")
                break

# ============================================================
# DISPLAY UPLOADED FILES WITH ❌
# ============================================================
if st.session_state.uploaded_images:
    st.markdown("### 📂 Uploaded Files")

    for i, f in enumerate(st.session_state.uploaded_images):
        col1, col2, col3 = st.columns([1, 4, 1])

        with col1:
            st.image(Image.open(f), width=80)

        with col2:
            st.markdown(f"**{i+1}. {f.name}**")

        with col3:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.uploaded_images.pop(i)
                st.rerun()

# ============================================================
# INFERENCE (ONLY WHEN EXACTLY 6 IMAGES)
# ============================================================
if len(st.session_state.uploaded_images) == 6:
    images = [Image.open(f) for f in st.session_state.uploaded_images]

    grid_image = make_image_grid_pil(images)
    st.divider()
    st.markdown("### 🧩 Stitched Input Grid")
    st.image(grid_image, width=900)

    with st.spinner("Running inference..."):
        x = transform(grid_image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]

    rgb = np.clip(pred[:3] * 255.0, 0, 255)
    sd  = np.clip(pred[3:] * sd_max, 0, sd_max)

    st.divider()
    st.markdown("### 📊 Predicted Coating Values")

    cols = st.columns(3)
    for c, v, l in zip(cols, rgb, ["R", "G", "B"]):
        c.markdown(f"<h2>{l}</h2><h1>{v:.1f}</h1>", unsafe_allow_html=True)

    cols = st.columns(3)
    for c, v, l in zip(cols, sd, ["SD_R", "SD_G", "SD_B"]):
        c.markdown(f"<h2>{l}</h2><h1>{v:.3f}</h1>", unsafe_allow_html=True)

    # ========================================================
    # PDF REPORT
    # ========================================================
    def create_pdf():
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4

        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(w/2, h-40, "CV Grain Analysis Report")

        c.setFont("Helvetica", 12)
        c.drawCentredString(w/2, h-65, f"Grain type: {grain_type.split(' ',1)[1]}")
        c.drawCentredString(w/2, h-85, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        img = ImageReader(grid_image)
        c.drawImage(img, 50, h-420, width=500, height=300, preserveAspectRatio=True)

        y = h - 460
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Predicted Values")
        y -= 30

        rows = [
            ("R", f"{rgb[0]:.2f}"),
            ("G", f"{rgb[1]:.2f}"),
            ("B", f"{rgb[2]:.2f}"),
            ("SD_R", f"{sd[0]:.3f}"),
            ("SD_G", f"{sd[1]:.3f}"),
            ("SD_B", f"{sd[2]:.3f}")
        ]

        c.setFont("Helvetica", 12)
        for k, v in rows:
            c.drawString(70, y, k)
            c.drawString(200, y, v)
            y -= 22

        c.showPage()
        c.save()
        buf.seek(0)
        return buf

    st.divider()
    st.download_button(
        "📄 Download PDF Report",
        data=create_pdf(),
        file_name=f"grain_analysis_{grain_type.split()[1].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )
