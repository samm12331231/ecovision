import streamlit as st
import requests

def upload_section():
    st.header("📤 Input")
    
    input_mode = st.radio("Input Type", ["Upload Image", "Camera"], horizontal=True)
    
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take a photo")
    
    baseline_file = st.file_uploader("Baseline image (optional)", type=["png", "jpg", "jpeg"])
    
    result = None
    image_data = None
    
    if uploaded_file is not None and st.button("🚀 Analyze", type="primary"):
        with st.spinner("Processing..."):
            files = {"image": uploaded_file.getvalue()}
            if baseline_file:
                files["baseline"] = baseline_file.getvalue()
            
            try:
                response = requests.post(
                    "http://localhost:8000/analyze",
                    files=files,
                    data={"mode": "full"}
                )
                result = response.json()
                image_data = uploaded_file
                st.success(f"Done in {result['processing_time_ms']}ms")
            except Exception as e:
                st.error(f"Backend error: {e}")
    
    return result, image_data