import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# Load YOLOv5 model
#@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Function to analyze the image and return detected components and the image with bounding boxes
def analyze_image(model, image):
    
    image_np = np.array(image)
    
    results = model(image_np)

    labels = results.names
    detected_labels = [labels[int(cls)] for cls in results.xyxy[0][:, -1].tolist()]
    detected_labels = list(set(detected_labels))  # Get distinct labels

    for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
        label = f'{labels[int(cls)]} {conf:.2f}'
        plot_one_box(box, image_np, label=label, color=(255, 0, 0), line_thickness=2)
    
    return detected_labels, Image.fromarray(image_np)

# Function to plot bounding boxes on the image
def plot_one_box(x, img, color=(255, 0, 0), label=None, line_thickness=2):

    x = [int(i) for i in x]
    c1, c2 = (x[0], x[1]), (x[2], x[3])
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        font_scale = max(0.5, line_thickness / 3)
        font_thickness = max(1, line_thickness // 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_pos = (c1[0], c1[1] - text_size[1] - 3)
        cv2.rectangle(img, (c1[0], c1[1] - text_size[1] - 10), (c1[0] + text_size[0], c1[1]), color, -1)  # Filled
        cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, [225, 255, 255], font_thickness, lineType=cv2.LINE_AA)

# Streamlit user interface
st.title("Object Detection App")
st.write("Upload an image and click 'Analyze Image' to detect objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Analyze Image"):
        model = load_model()
        detected_components, detected_image = analyze_image(model, image)
        
        st.image(detected_image, caption='Detected Image', use_column_width=True)
        st.write("Detected Components:")
        for component in detected_components:
            st.write(component)
