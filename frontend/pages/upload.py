import streamlit as st
import requests
import io

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
            # Proper multipart format for requests
            files = [("image", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))]
            if baseline_file:
                files.append(("baseline", (baseline_file.name, baseline_file.getvalue(), baseline_file.type)))
            
            try:
                response = requests.post(
                    "http://localhost:8000/analyze",
                    files=files,
                    data={"mode": "full"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Pass BytesIO so dashboard can do Image.open()
                    image_data = io.BytesIO(uploaded_file.getvalue())
                    st.success(f"✅ Done in {result['processing_time_ms']}ms")
                else:
                    st.error(f"Backend returned {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is `uvicorn backend.main:app --port 8000` running?")
            except Exception as e:
                st.error(f"Error: {e}")
    
    return result, image_data