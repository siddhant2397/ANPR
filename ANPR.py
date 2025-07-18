import streamlit as st
import pandas as pd
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import numpy as np
import requests
import re
import io

# --- CONFIG ---
ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
MINDEE_API_KEY = st.secrets["MINDEE_API_KEY"]
ROBOFLOW_MODEL_ID = "numberplate_data_v1/1"  # Update as needed

st.title("Full ANPR: Roboflow Detection + Mindee Cloud OCR + Authorization Check")

# --- EXCEL/CSV: Upload and Parse Authorized Plates List ---
auth_file = st.file_uploader("Upload Excel/CSV of Authorized Plates", type=["xlsx", "xls", "csv"])

if auth_file:
    if auth_file.name.endswith(".csv"):
        df = pd.read_csv(auth_file)
    else:
        df = pd.read_excel(auth_file)
    plate_col = df.columns[0]  # assumes first column
    authorized_plates = set(
        re.sub(r'[^A-Za-z0-9]', '', str(x)).upper()
        for x in df[plate_col].dropna()
    )
    st.success(f"{len(authorized_plates)} authorized plate(s) loaded.")
else:
    authorized_plates = set()
    st.info("Upload your authorized plate Excel/CSV before uploading images.")

# --- Roboflow Detection Client ---
CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=ROBOFLOW_API_KEY)

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

def extract_plate_mindee(image_crop: Image.Image) -> str:
    """Crop PIL image, send to Mindee API, return normalized string"""
    api_url = "https://api.mindee.net/v1/products/mindee/license-plate/v1/predict"
    buf = io.BytesIO()
    image_crop.save(buf, format="JPEG")
    buf.seek(0)

    headers = {
        "Authorization": f"Token {MINDEE_API_KEY}"
    }
    files = {"document": ("crop.jpg", buf, "image/jpeg")}
    response = requests.post(api_url, headers=headers, files=files)
    if response.status_code == 200:
        res = response.json()
        # Mindee's number is in "license_plates" > "value" > text (normalize)
        plats = res.get("document", {}).get("inference", {}).get("license_plates", [])
        if plats:
            plate_val = plats[0]["value"]
            return re.sub(r'[^A-Za-z0-9]', '', plate_val).upper()
    return ""

if uploaded_file and authorized_plates:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)
    image.save("temp_upload.jpg")  # for Roboflow API

    with st.spinner("Detecting plates..."):
        result = CLIENT.infer("temp_upload.jpg", model_id=ROBOFLOW_MODEL_ID)

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

        # --- CLOUD OCR WITH MINDEE ---
        plate_crop = image.crop((x0, y0, x1, y1))
        plate_text = extract_plate_mindee(plate_crop)

        if plate_text:
            plates.append(plate_text)
            if plate_text in authorized_plates:
                status = "✅ AUTHORIZED"
                color = "green"
            else:
                status = "❌ UNAUTHORIZED"
                color = "red"

            # Annotate image and result
            drawer.text((x0, y0-15), f"{plate_text} ({status})", fill=color)
            st.markdown(
                f"<span style='color:{color}'><b>Plate {i}: {plate_text} — {status}</b></span>",
                unsafe_allow_html=True,
            )
        else:
            st.write(f"Plate {i}: Unable to detect text (Mindee)")

    st.image(image_draw, caption="Detected Plates & Authorization", use_container_width=True)
    st.caption("Bounding boxes, detected numbers, and authorization status.")
    st.expander("Detection Data").write(detections)
