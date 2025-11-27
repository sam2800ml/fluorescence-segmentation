import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors # <--- NEW IMPORT
from model_arch import AttentionUNet, Unet, NestedUNet # Assuming these are defined elsewhere
import os
import io
import matplotlib.pyplot as plt


# ============ CORRECTED INPUT HANDLING ============
def prepare_input_image(uploaded_file, input_mode="auto"):
    """
    Handle both single-channel and multi-channel inputs correctly.
    
    Args:
        uploaded_file: Streamlit file uploader object
        input_mode: "auto" (detect), "single" (grayscale), or "combined" (3-channel RGB)
    
    Returns:
        img_resized: (512, 512, 3) normalized image ready for model
    """
    img = Image.open(uploaded_file).convert("RGB") # Explicitly convert to RGB initially
    img_array = np.array(img)
    
    # Handle single-channel or multi-channel
    if len(img_array.shape) == 2:  # Grayscale
        # Stack to 3 channels (for model expecting RGB)
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:  # Single channel with explicit dimension
        img_array = np.repeat(img_array, 3, axis=-1)
    elif img_array.shape[2] != 3:  # Not RGB (e.g., RGBA)
        # Convert to RGB (already done above, but good safeguard)
        img_array = img_array[..., :3] 

    # Resize
    img_resized = cv2.resize(img_array, (512, 512))
    
    # Ensure uint8
    if img_resized.dtype != np.uint8:
        if img_resized.max() <= 1.0:
            img_resized = (img_resized * 255).astype(np.uint8)
        else:
            img_resized = img_resized.astype(np.uint8)
    
    return img_resized


def normalize_image(img_resized):
    """ImageNet normalization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized / 255.0 - mean) / std
    return img_norm


# ============ CORRECTED COLORMAP RENDERING ============
def render_jet_mask(pred_mask, num_classes=4):
    """
    Convert class indices to jet colormap using Matplotlib's Boundary Norm
    to ensure classes map correctly to discrete colors like in Colab's imshow.
    """
    # 1. Get the 'jet' colormap, discretized into num_classes
    cmap = cm.get_cmap("jet", num_classes)
    
    # 2. Define boundaries to correctly map integers (0, 1, 2, 3)
    # Boundaries are set at -0.5, 0.5, 1.5, 2.5, 3.5
    bounds = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # 3. Apply colormap and scale to 0-255
    pred_rgba = cmap(norm(pred_mask))  # (H, W, 4)
    pred_color = (pred_rgba[..., :3] * 255).astype(np.uint8)  # (H, W, 3)
    
    return pred_color


# ============ STREAMLIT UI ============
st.set_page_config(page_title="Cell Segmentation Demo", page_icon="🔬", layout="wide")
st.title("🔬 Fluorescence Cell Segmentation")
st.markdown("Upload protein-only or combined images – model predicts 4 classes (bg / nucleus / cytoplasm / protein).")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Model Settings")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"🚀 Device: {device}")

    model_type = st.selectbox(
        "Select Model Type",
        ["Protein Model (protein input → 4 classes)", "Combined Model (full input → 4 classes)"]
    )

    if model_type.startswith("Protein"):
        MODEL_PATH = "/app/frontpage/unet_protein_model_.pth"
        NUM_CLASSES = 4
        MODEL_ARCH = Unet
        st.info("🔬 Protein model: trained on protein channel only.")
    else:
        MODEL_PATH = "/app/frontpage/NestedUNet_comb_model.pth"
        NUM_CLASSES = 4
        MODEL_ARCH = NestedUNet
        st.info("🧬 Combined model: full 3-channel input (nucleus, cytoplasm, protein).")

    @st.cache_resource
    def load_model(_model_path, _num_classes, _model_arch):
        try:
            if not os.path.exists(_model_path):
                st.error(f"❌ Model missing: {_model_path}")
                return None

            model = _model_arch(in_channels=3, num_classes=_num_classes)
            checkpoint = torch.load(_model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            st.success(f"✅ Loaded model: {model_type}")
            return model
        except Exception as e:
            st.error(f"Load error: {e}")
            return None

    model = load_model(MODEL_PATH, NUM_CLASSES, MODEL_ARCH)


# Main inference
if model is None:
    st.warning("⚠️ Fix model loading first.")
    st.stop()

uploaded_file = st.file_uploader("Upload fluorescence image…", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # ============ FIXED INPUT PROCESSING ============
        img_resized = prepare_input_image(uploaded_file)
        img_norm = normalize_image(img_resized)

        img_tensor = (
            torch.tensor(img_norm)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # Prediction
        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        # ============ FIXED COLORMAP RENDERING ============
        # The fix is in this function call's implementation
        pred_color = render_jet_mask(pred_mask, NUM_CLASSES)

        # Display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(img_resized, channels="RGB", use_container_width=True)

        with col2:
            st.subheader("Prediction (JET colormap)")
            st.image(pred_color, channels="RGB", use_container_width=True)
            
            # Class legend
            st.caption("""
            **Color mapping:**
            - 🔵 Dark Blue → Background (0)
            - 🔵 Cyan → Nucleus (1)
            - 🟡 Yellow → Cytoplasm (2)
            - 🔴 Red → Protein (3)
            """)

        # Overlay
        st.subheader("Overlay (input + prediction)")
        # Ensure pred_color is (512, 512, 3) and uint8 before overlay
        overlay = cv2.addWeighted(img_resized, 0.6, pred_color, 0.4, 0)
        st.image(overlay, channels="RGB", use_container_width=True)

        # Class distribution
        unique, counts = np.unique(pred_mask, return_counts=True)
        class_names = ["Background (0)", "Nucleus (1)", "Cytoplasm (2)", "Protein (3)"]
        class_dist = {class_names[i]: int(c) for i, c in zip(unique, counts)}
        
        st.subheader("Class Distribution")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.json(class_dist)
        with col_dist2:
            # Simple bar chart
            # import matplotlib.pyplot as plt # Already imported above
            fig, ax = plt.subplots(figsize=(6, 4))
            names = [class_names[u] for u in unique]
            ax.bar(names, counts)
            ax.set_ylabel("Pixel Count")
            ax.set_title("Class Distribution")
            st.pyplot(fig)

        # Downloads
        st.divider()
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        def to_png_bytes(ndarr):
            buf = io.BytesIO()
            Image.fromarray(ndarr).save(buf, format="PNG")
            return buf.getvalue()

        with col_dl1:
            st.download_button(
                "📥 Jet mask PNG",
                data=to_png_bytes(pred_color),
                file_name="prediction_jet.png",
                mime="image/png",
            )

        with col_dl2:
            st.download_button(
                "📥 Raw mask (grayscale)",
                data=to_png_bytes(pred_mask.astype(np.uint8) * 85),  # Scale 0-3 to 0-255
                file_name="prediction_raw.png",
                mime="image/png",
            )

        with col_dl3:
            st.download_button(
                "📥 Input image",
                data=uploaded_file.getvalue(),
                file_name="input.png",
                mime="image/png",
            )

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")
        st.write(f"Details: {type(e).__name__}")

else:
    st.info("📤 Upload an image to see predictions.")