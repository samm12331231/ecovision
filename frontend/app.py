cd ~/ecovision
mkdir -p frontend
cat > frontend/app.py << 'PYEOF'
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="UAE EcoVision", layout="wide")

st.title("🌿 UAE EcoVision")
st.subheader("Environmental Monitoring System")

# Sidebar
st.sidebar.header("Input")
input_mode = st.sidebar.radio("Input Type", ["Upload Image", "Camera"])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload aerial image", type=["png", "jpg", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take a photo")
    
    baseline_file = st.file_uploader("Upload baseline image (optional)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None and st.button("Analyze"):
        with st.spinner("Processing..."):
            # Prepare files for API
            files = {"image": uploaded_file.getvalue()}
            if baseline_file:
                files["baseline"] = baseline_file.getvalue()
            
            try:
                # Call backend API
                response = requests.post(
                    "http://localhost:8000/analyze",
                    files=files,
                    data={"mode": "full"}
                )
                result = response.json()
                
                # Store result in session state for dashboard
                st.session_state.result = result
                st.session_state.image = uploaded_file
                
                st.success(f"Done in {result['processing_time_ms']}ms")
                
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

with col2:
    st.header("Results")
    
    if "result" in st.session_state:
        result = st.session_state.result
        
        # Display uploaded image
        if "image" in st.session_state:
            img = Image.open(st.session_state.image)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Show results
        if "segmentation" in result["results"]:
            score = result["results"]["segmentation"]["health_score"]
            st.metric("Vegetation Health Score", f"{score:.1f}%")
        
        if "detection" in result["results"]:
            num = result["results"]["detection"]["num_objects"]
            st.metric("Objects Detected", num)
        
        # Raw JSON (for debugging)
        with st.expander("Raw API Response"):
            st.json(result)
        
        # Export button
        if st.button("Export Report"):
            report_text = str(result)
            st.download_button(
                "Download Report",
                report_text,
                file_name="ecovision_report.txt"
            )
    else:
        st.info("Upload an image and click Analyze to see results")
PYEOF