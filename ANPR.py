import streamlit as st
from inference import get_model
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import os
import easyocr

os.environ["ROBOFLOW_API_KEY"] = "trSfqkdVMHiCilTnndyy"  # Replace with your API key

model = get_model(model_id="numberplate_data_v1/1")  # Replace with your model ID
ocr_reader = easyocr.Reader(['en'])

st.title("Roboflow ANPR with OCR")

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Input Image", use_container_width=True)
    image_np = np.array(image_pil)

    with st.spinner("Detecting license plates..."):
        results = model.infer(image_np)[0]
        detections = sv.Detections.from_inference(results)
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(scene=image_np, detections=detections)
        st.image(annotated, caption="Detected Plates", use_container_width=True)

        st.subheader("Recognition Results:")
        if not detections.is_empty:
            for idx, xyxy in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, xyxy)
                plate_crop = image_np[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    st.write(f"Detection {idx+1}: Unable to crop plate region.")
                    continue
                # OCR on cropped plate
                ocr_result = ocr_reader.readtext(plate_crop)
                if ocr_result:
                    text = ocr_result[0][1]
                    st.write(f"Plate #{idx+1}: **{text}**")
                else:
                    st.write(f"Plate #{idx+1}: OCR failed to detect text.")
        else:
            st.write("No plates detected.")
