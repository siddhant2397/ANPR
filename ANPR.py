import streamlit as st
import pandas as pd
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import numpy as np
from paddleocr import PaddleOCR
import cv2
import os
import re

# Load secrets and model info
API_KEY = st.secrets["ROBOFLOW_API_KEY"]    # Set as Streamlit secret
MODEL_ID = "numberplate_data_v1/1"          # Your Roboflow model ID

st.title("Full ANPR: Roboflow Detection + EasyOCR + Authorization Check")

# Upload Excel or CSV file with authorized plate numbers
auth_file = st.file_uploader("Upload Excel/CSV of Authorized Plates", type=["xlsx", "xls", "csv"])

if auth_file:
    # Read plate numbers from file
    if auth_file.name.endswith(".csv"):
        df = pd.read_csv(auth_file)
    else:
        df = pd.read_excel(auth_file)
    # Try to detect which column holds the plate numbers
    plate_col = df.columns[0]  # assumes first column
    authorized_plates = set(
    re.sub(r'[^A-Za-z0-9]', '', str(x)).upper()
    for x in df[plate_col].dropna()
)

    st.success(f"{len(authorized_plates)} authorized plate(s) loaded.")
else:
    authorized_plates = set()
    st.info("Please upload your authorized plate list Excel/CSV before uploading images.")

# Usual ANPR pipeline
CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file and authorized_plates:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)
    image.save("temp_upload.jpg")  # For Roboflow API

    with st.spinner("Detecting plates..."):
        result = CLIENT.infer("temp_upload.jpg", model_id=MODEL_ID)

    detections = result.get("predictions", [])
    plates = []

    image_draw = image.copy()
    drawer = ImageDraw.Draw(image_draw)
    st.subheader("Detection, OCR & Authorization Results")

    for i, det in enumerate(detections, 1):
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        x0, y0 = int(x - w/2), int(y - h/2)
        x1, y1 = int(x + w/2), int(y + h/2)
        drawer.rectangle([x0, y0, x1, y1], outline="red", width=2)

        plate_crop = image.crop((x0, y0, x1, y1))
        plate_np = np.array(plate_crop)
        if plate_np.shape[2] == 3:
            plate_bgr = cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)
        else:
            plate_bgr = plate_np

        ocr_out = ocr_reader.readtext(plate_np)
        if ocr_out:
            if ocr_out and len(ocr_out[0]):
                raw_plate_text = ocr_out[0][0][1][0]
                plate_text = re.sub(r'[^A-Za-z0-9]', '', raw_plate_text).upper()
                plates.append(plate_text)
            else:
                plate_text = ""
            
            # Authorization check below!
            if plate_text in authorized_plates:
                status = "✅ AUTHORIZED"
                color = "green"
            else:
                status = "❌ UNAUTHORIZED"
                color = "red"

            # Annotate on image and print result
            drawer.text((x0, y0-15), f"{plate_text} ({status})", fill=color)
            st.markdown(f"<span style='color:{color}'><b>Plate {i}: {plate_text} — {status}</b></span>", unsafe_allow_html=True)
        else:
            st.write(f"Plate {i}: Unable to detect text")

    st.image(image_draw, caption="Detected Plates & Authorization", use_container_width=True)
    st.caption("Bounding boxes, detected numbers, and authorization status.")

    # Optional: Show detection JSON
    st.expander("Detection Data").write(detections)
