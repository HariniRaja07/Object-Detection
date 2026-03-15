import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("Object Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    # Convert image to RGB
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)

    st.image(image, caption="Original Image", width=600)

    # Load YOLOv8 medium model (better for detecting small objects)
    model = YOLO("yolov8m.pt")

    # Run detection on the uploaded image
    results = model(image, conf=0.25)  # Lower threshold detects more objects

    # Copy image for drawing
    annotated_image = image.copy()

    # Iterate over all detected boxes
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]  # Get object name

                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label with object name + confidence
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(annotated_image, caption="Detected Objects", width=600)
