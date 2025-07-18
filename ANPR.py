import streamlit as st
from mindee import ClientV2, InferenceParameters
from PIL import Image
import tempfile
import os
import json

# You should NOT hard-code API keys in production. Use Streamlit secrets!
api_key = st.secrets["MINDEE_API_KEY"]
model_id = "7889f4de-4ddb-4fd9-9fa4-270f24a670de"  # Replace as needed

st.title("Mindee Custom Model Inference")

uploaded_file = st.file_uploader(
    "Upload an image or PDF for inference",
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1]  # Get user's original extension, e.g., ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name  # Now has correct extension!

    st.info(f"Temp file size: {os.path.getsize(input_path)} bytes")


    # Initialize Mindee client and parameters
    mindee_client = ClientV2(api_key)
    params = InferenceParameters(
        model_id=model_id,
        rag=False,
    )
    input_source = mindee_client.source_from_path(input_path)
    
    with st.spinner("Running inference..."):
        try:
            response = mindee_client.enqueue_and_get_inference(input_source, params)
            st.success("Inference complete!")
            # Print main result as formatted JSON
            # Temporary debug lines
            # Show entire parsed API result:
            if isinstance(response.raw_http, str):
                data = json.loads(response.raw_http)
            else:
                data = response.raw_http



# Show just the plate number (robustly):
            plate_val = (
                response.raw_http.get("inference", {})
                .get("result", {})
                .get("fields", {})
                .get("license_plate", {})
                .get("value", None)
)

            if plate_val:
                st.success(f"License Plate Number: {plate_val}")
            else:
                st.warning("No license plate detected.")

            
        except Exception as e:
            st.error(f"Inference failed: {e}")

    # Delete the temp file
    os.remove(input_path)
