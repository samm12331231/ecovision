import streamlit as st
from PIL import Image

def dashboard_section(result, image_data):
    st.header("📊 Results")
    
    if result is None:
        st.info("Upload an image and click Analyze to see results")
        return
    
    if image_data:
        img = Image.open(image_data)
        st.image(img, caption="Original Image", use_column_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = result["results"].get("segmentation", {}).get("health_score", 0)
        st.metric("🌿 Health", f"{score:.1f}%")
    
    with col2:
        num = result["results"].get("detection", {}).get("num_objects", 0)
        st.metric("📦 Objects", num)
    
    with col3:
        changed = result["results"].get("change", {}).get("changed_pixels", 0)
        st.metric("🔥 Changes", changed)
    
    with col4:
        edges = result["results"].get("edges", {}).get("edge_pixels", 0)
        st.metric("📐 Edges", edges)
    
    st.divider()
    report_text = str(result)
    st.download_button(
        "📥 Export Report",
        report_text,
        file_name="ecovision_report.txt",
        mime="text/plain"
    )