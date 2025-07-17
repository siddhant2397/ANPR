import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

# Load YOLO model (replace with your custom-trained plate weights if available)
plate_model = YOLO('yolov8n.pt')  
ocr_reader = easyocr.Reader(['en'])

st.title("Automatic Number Plate Recognition (ANPR) App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert('RGB')
    st.image(pil_img, caption="Uploaded Vehicle Image", use_container_width=True)
    frame = np.array(pil_img)
    with st.spinner("Detecting license plates..."):
        # Run YOLO detection
        results = plate_model(frame)
        plates = []
        # Go through each detected plate
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                plate_img = frame[y1:y2, x1:x2]
                st.image(plate_img, caption="Detected Plate", use_container_width=True)
                # OCR on cropped plate region
                ocr_result = ocr_reader.readtext(plate_img)
                if ocr_result:
                    text = ocr_result[0][1]
                    plates.append(text)
        st.header("Detected Plate(s):")
        if plates:
            for pt in plates:
                st.write(f"**{pt}**")
        else:
            st.write("No plates detected or OCR failed.")
