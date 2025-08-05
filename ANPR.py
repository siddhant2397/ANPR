import streamlit as st
from mindee import ClientV2, InferenceParameters
from PIL import Image
import tempfile
import os
import json
import pandas as pd
import re

# You should NOT hard-code API keys in production. Use Streamlit secrets!
api_key = st.secrets["MINDEE_API_KEY"]
model_id = "863681b0-83bd-4de6-88cb-693906104892"  # Replace as needed

st.title("Automatic Number Plate Recognition (Developed by Siddhant Goswami")
auth_file = st.file_uploader("Upload Excel/CSV of Authorized Plates", type=["xlsx", "xls", "csv"])
if auth_file is not None:
    # Read authorized plates from file
    if auth_file.name.endswith(".csv"):
        df = pd.read_csv(auth_file)
    else:
        df = pd.read_excel(auth_file)
    plate_col = df.columns[0]  # assumes first column is plate numbers
    # Uniform: strip, uppercase, remove all non-alphanum
    authorized_plates = set(
        re.sub(r'[^A-Za-z0-9]', '', str(x)).upper()
        for x in df[plate_col].dropna()
    )
    st.success(f"{len(authorized_plates)} authorized plate(s) loaded.")
else:
    authorized_plates = set()
    st.info("Please upload your authorized plate Excel/CSV before uploading images.")


uploaded_file = st.file_uploader(
    "Upload an image or PDF for inference",
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None and authorized_plates:
    file_ext = os.path.splitext(uploaded_file.name)[1]  # Get user's original extension, e.g., ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name  # Now has correct extension!

    st.info(f"Temp file size: {os.path.getsize(input_path)} bytes")
    if file_ext.lower() in [".jpg", ".jpeg", ".png"]:
        try:
            img = Image.open(input_path)
            st.image(img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.warning(f"Unable to display image preview: {e}")
    elif file_ext.lower() == ".pdf":
        st.info("PDF uploaded. Image preview not available, but inference will run.")


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
                data.get("inference", {})
                .get("result", {})
                .get("fields", {})
                .get("license_plate", {})
                .get("value", None)
)

            if plate_val:
                plate_val_uniform = re.sub(r'[^A-Za-z0-9]', '', plate_val).upper()
                st.success(f"License Plate Number: {plate_val}")
                # Compare with authorized list
                if plate_val_uniform in authorized_plates:
                    st.success("✅ AUTHORIZED")
                else:
                    st.error("❌ UNAUTHORIZED")

            else:
                st.warning("No license plate detected.")

            
        except Exception as e:
            st.error(f"Inference failed: {e}")

    # Delete the temp file
    os.remove(input_path)
