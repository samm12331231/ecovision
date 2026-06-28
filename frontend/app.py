import streamlit as st
import sys
from pathlib import Path

# Add project root to path so 'frontend' and 'backend' are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.pages.upload import upload_section
from frontend.pages.dashboard import dashboard_section
from frontend.live import live_section

st.set_page_config(page_title="UAE EcoVision", layout="wide", page_icon="🌿")

st.title("🌿 UAE EcoVision")
st.subheader("Environmental Monitoring System")

# Input modality selector: still image analysis (the original workflow)
# or live/uploaded video processing. The image flow below is unchanged.
mode = st.radio(
    "Mode", ["🖼️ Image Analysis", "🎥 Live Video"],
    horizontal=True, label_visibility="collapsed",
)

if mode == "🖼️ Image Analysis":
    # Two columns
    left, right = st.columns(2)

    with left:
        result, image_data = upload_section()

    with right:
        dashboard_section(result, image_data)
else:
    live_section()