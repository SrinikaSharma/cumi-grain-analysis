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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CV Grain Analysis",
    page_icon="🔬",
    layout="centered"
)

st.markdown(
    """
    <style>
    /* Hide file list inside file uploader */
    div[data-testid="stFileUploader"] ul {
        display: none;
    }

    /* Hide pagination text (page 1 of 2) */
    div[data-testid="stFileUploader"] > div > div:last-child {
        display: none;
    }
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
# CHECKPOINT + SD_MAX ROUTING
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

st.success(
    f"✔ Loaded model for **{grain_type.replace('🔴 ', '').replace('⚫ ', '').replace('🟣 ', '').replace('⚪ ', '')}** "
)

st.divider()

# ============================================================
# IMAGE GRID FUNCTION
# ============================================================
def make_image_grid_pil(images):
    assert len(images) == 6, "Exactly 6 images are required"

    grid_width, grid_height = 900, 600
    cols, rows = 3, 2
    cell_w, cell_h = grid_width // cols, grid_height // rows

    grid = Image.new("RGB", (grid_width, grid_height))
    for i, img in enumerate(images):
        img = img.convert("RGB").resize((cell_w, cell_h), Image.BILINEAR)
        grid.paste(img, ((i % cols) * cell_w, (i // cols) * cell_h))

    return grid

# ============================================================
# MODEL DEFINITION (MATCHES CHECKPOINT)
# ============================================================
class ResNetDualHead(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.rgb_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.sd_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        feats = self.backbone(x)
        rgb = self.rgb_head(feats)
        sd = self.sd_head(feats)
        return torch.cat([rgb, sd], dim=1)

# ============================================================
# LOAD MODEL (CACHED, CHECKPOINT-AWARE)
# ============================================================
@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetDualHead(pretrained=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

model, device = load_model(CHECKPOINT_PATH)

# ============================================================
# TRANSFORMS
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
# STEP 1 — IMAGE UPLOAD
# ============================================================
st.markdown("### 📤 Step 1: Upload Exactly 6 Images")

uploaded = st.file_uploader(
    "Upload 6 grain images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded:
    st.markdown("### 📂 Uploaded files")

    for i, file in enumerate(uploaded, start=1):
        col1, col2 = st.columns([1, 4])
        with col1:
            img = Image.open(file)
            st.image(img, width=80)
        with col2:
            st.markdown(f"**{i}. {file.name}**")
st.divider()

if uploaded:
    if len(uploaded) != 6:
        st.error("❌ Exactly 6 images are required.")
        st.stop()

    images = [Image.open(f) for f in uploaded]

    

    # ========================================================
    # STITCHED GRID
    # ========================================================
    st.markdown("### 🧩 Stitched Input Grid (900 × 600)")
    grid_image = make_image_grid_pil(images)
    st.image(grid_image, width=900)

    # ========================================================
    # INFERENCE
    # ========================================================
    with st.spinner("Running inference..."):
        x = transform(grid_image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]

    rgb = np.clip(pred[:3] * 255.0, 0.0, 255.0)
    sd = np.clip(pred[3:] * sd_max, 0.0, sd_max)

    st.divider()

    # ========================================================
    # RESULTS
    # ========================================================
    st.markdown("### 📊 Predicted Coating Values")

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .metric-title {
            font-size: 30px;
            font-weight: 600;
            color: #555;
        }
        .metric-value {
            font-size: 30px;
            font-weight: 600;
            color: #111;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    "<p style='font-size:28px; font-weight:600;'>"
    "R, G, B values of the displayed image:"
    "</p>",
    unsafe_allow_html=True
    )


    # RGB ROW
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">R</div>
                <div class="metric-value">{rgb[0]:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">G</div>
                <div class="metric-value">{rgb[1]:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">B</div>
                <div class="metric-value">{rgb[2]:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    "<p style='font-size:28px; font-weight:600; margin-top:20px;'>"
    "Standard Deviations of R, G, B in displayed image:"
    "</p>",
    unsafe_allow_html=True
    )


    # SD ROW
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">SD_R</div>
                <div class="metric-value">{sd[0]:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">SD_G</div>
                <div class="metric-value">{sd[1]:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">SD_B</div>
                <div class="metric-value">{sd[2]:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

