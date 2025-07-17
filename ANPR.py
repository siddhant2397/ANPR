import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import numpy as np
import easyocr
import os

# Securely load your secrets: Set this in Streamlit Cloud > Settings > Secrets
API_KEY = st.secrets["ROBOFLOW_API_KEY"]    # NEVER hard-code sensitive info!
MODEL_ID = "numberplate_data_v1/1"          # Replace with your model ID

st.title("Full ANPR: Roboflow Detection + EasyOCR Number Recognition")

# Initialize clients/readers once
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)
ocr_reader = easyocr.Reader(['en'])

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)
    image.save("temp_upload.jpg")  # Save for Roboflow call

    with st.spinner("Detecting plates (via Roboflow API)..."):
        result = CLIENT.infer("temp_upload.jpg", model_id=MODEL_ID)

    detections = result.get("predictions", [])
    plates = []

    # Draw bboxes and OCR results
    image_draw = image.copy()
    drawer = ImageDraw.Draw(image_draw)
    st.subheader("Detection and OCR Results")

    for i, det in enumerate(detections, 1):
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        x0, y0 = int(x - w/2), int(y - h/2)
        x1, y1 = int(x + w/2), int(y + h/2)
        # Draw the detection box
        drawer.rectangle([x0, y0, x1, y1], outline="red", width=2)

        # Crop plate for OCR
        plate_crop = image.crop((x0, y0, x1, y1))
        plate_np = np.array(plate_crop)

        # OCR the cropped region
        ocr_out = ocr_reader.readtext(plate_np)
        if ocr_out:
            plate_text = ocr_out[0][1]
            plates.append(plate_text)
            # Annotate on the output image
            drawer.text((x0, y0-15), plate_text, fill="red")
            st.write(f"**Plate {i}: `{plate_text}`**")
        else:
            st.write(f"*Plate {i}: Unable to detect text*")

    st.image(image_draw, caption="Detected Plates & OCR", use_container_width=True)
    st.caption("Bounding boxes and text are drawn on the uploaded image.")

    # Optionally, display raw detection data
    st.expander("Detection JSON").write(detections)

    # Clean up temp file if desired
    # os.remove("temp_upload.jpg")
