import streamlit as st
import requests
import numpy as np
import cv2
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


API_URL = "http://inference-api:8000/predict"


def render_jet_mask(pred_mask, num_classes=4):
    """Consistent colormap rendering for segmentation."""
    cmap = cm.get_cmap("jet", num_classes)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    pred_rgba = cmap(norm(pred_mask))
    return (pred_rgba[..., :3] * 255).astype(np.uint8)

def to_png_bytes(image_array):
    """Helper for download buttons."""
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    if is_success:
        return buffer.tobytes()
    return None


st.set_page_config(page_title="Cell Segmentation", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Fluorescence Cell Segmentation")
st.sidebar.header("📊 Model Info")
st.sidebar.info("Model: NestedUNet\nClasses: 4 (BG, Nuclei, Cyto, Protein)")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    main_col, side_col = st.columns([3, 1])

    with main_col:
 
        with st.spinner("🧠 Inference in progress..."):
            try:
      d
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=1000)
                
                if response.status_code == 200:
                    result = response.json()
                    
      
                    mask_bytes = base64.b64decode(result["mask_base64"])
                    nparr = np.frombuffer(mask_bytes, np.uint8)
                    pred_mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    
        
                    img = Image.open(uploaded_file).convert("RGB")
                    img_resized = np.array(img.resize((512, 512)))
                    pred_color = render_jet_mask(pred_mask)
                    overlay = cv2.addWeighted(img_resized, 0.7, pred_color, 0.3, 0)

                    tab1, tab2, tab3 = st.tabs(["🖼️ Side-by-Side", "🎭 Overlay", "📈 Statistics"])
                    
                    with tab1:
                        c1, c2 = st.columns(2)
                        c1.image(img_resized, caption="Original", use_container_width=True)
                        c2.image(pred_color, caption="Segmentation", use_container_width=True)
                    
                    with tab2:
                        st.image(overlay, caption="70/30 Alpha Blend Overlay", use_container_width=True)
                    
                    with tab3:
                        dist = result["class_distribution"]
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(dist.keys(), dist.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                        ax.set_title("Pixel Count per Class")
                        st.pyplot(fig)

                elif response.status_code == 503:
                    st.error("⚠️ The Model is still loading on the server. Please wait 30 seconds and try again.")
                else:
                    st.error(f"❌ Server Error: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. The server is taking too long to process the image.")
            except Exception as e:
                st.error(f"📡 Connection Error: Ensure backend container is running. ({e})")

    with side_col:
        st.subheader("📥 Actions")
        if 'pred_color' in locals():
            st.download_button(
                "Download Mask (PNG)",
                data=to_png_bytes(pred_color),
                file_name="mask.png",
                mime="image/png"
            )
            st.download_button(
                "Download Overlay",
                data=to_png_bytes(overlay),
                file_name="overlay.png",
                mime="image/png"
            )
        
        st.divider()
        st.markdown("**Legend:**")
        st.markdown("🔵 Dark Blue: Background\n\n🟢 Green/Cyan: Nucleus\n\n🟡 Yellow: Cytoplasm\n\n🔴 Red: Protein")

else:
    st.info("👋 Welcome! Please upload a fluorescence image in the sidebar to begin.")