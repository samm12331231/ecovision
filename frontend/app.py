import streamlit as st
import sys
from pathlib import Path

# Add pages folder to path
sys.path.insert(0, str(Path(__file__).parent / "pages"))

from upload import upload_section
from dashboard import dashboard_section

st.set_page_config(page_title="UAE EcoVision", layout="wide", page_icon="🌿")

st.title("🌿 UAE EcoVision")
st.subheader("Environmental Monitoring System")

# Two columns
left, right = st.columns(2)

with left:
    result, image_data = upload_section()

with right:
    dashboard_section(result, image_data)