import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import time
import cv2
from PIL import Image
import numpy as np

# ----------------------------#
# App Configuration
# ----------------------------#
st.set_page_config(
    page_title="YOLO Object Detection - SH17 Dataset",
    page_icon="🎯",
    layout="wide",
)

# ----------------------------#
# Load Model
# ----------------------------#
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Replace with your custom model if available
    return model

model = load_model()

# ----------------------------#
# Helper Functions
# ----------------------------#
def detect_objects(uploaded_file, source_type="image"):
    """
    Run YOLO detection on an image or video.
    """
    if source_type == "image":
        img = Image.open(uploaded_file)
        results = model(img, conf=0.5)
        res_img = results[0].plot()
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        st.image(res_img, caption="Detected Objects", width="content")
        st.success("✅ Detection complete!")

    elif source_type == "video":
        st.info("🎥 Running detection on video...")
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        results = model.predict(source=video_path, stream=True, conf=0.5)
        stframe = st.empty()

        for i, r in enumerate(results):
            if i % 5 == 0:  # show every 5th frame for smoother display
                frame = r.plot()
                stframe.image(frame, channels="BGR", use_container_width=True)
        st.success("✅ Video detection complete!")

    elif source_type == "webcam":
        st.info("🎥 Starting live detection... Press Stop to end.")
        run = st.checkbox("▶️ Run Webcam", value=True)
        cap = cv2.VideoCapture(1)
        stframe = st.empty()

        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("No camera feed detected.")
                break

            results = model.predict(frame, conf=0.5)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            time.sleep(0.001)  # small delay to avoid CPU overload

        cap.release()
        st.success("✅ Live detection ended.")

# ============================================

# ----------------------------#
# Navigation
# ----------------------------#
pages = ["🏠 Home", "📖 Project Overview", "🧩 Try the Model", "ℹ️ About"]
page = st.sidebar.radio("Navigate", pages)

# ----------------------------#
# HOME PAGE
# ----------------------------#
if page == "🏠 Home":
    st.title("🎯 YOLO Object Detection on SH17 Dataset")
    st.subheader("Smart Vision — Real-Time Detection Using YOLO and SH17 Data")

    st.image("sample (1).png",
             caption="Sample detection showing bounding boxes using YOLO", 
             use_container_width=True)
    
    st.markdown("""
    Welcome to the **YOLO Object Detection Application**, a deep learning-powered tool for detecting multiple objects in real time.
    
    This project integrates **Ultralytics YOLOv8** with the **SH17 Dataset**, designed to train AI models that can accurately recognize various object categories in diverse environments.

    ---
    ### 🎯 Purpose
    The objective is to demonstrate the **application of AI in visual perception**, allowing automated systems to identify and classify multiple objects simultaneously.
    
    The project can be applied in:
    - Smart surveillance systems  
    - Road safety and traffic monitoring  
    - Agricultural yield analysis  
    - Industrial automation  
    - Academic and research-oriented projects  

    ---
    ### 🌟 Promoting Data Intelligence
    This initiative is championed by **Philip Code Academy**, with a mission to cultivate analytical minds capable of building data-driven, AI-enabled solutions.

    """)

    st.markdown("<br><hr><center>© 2025 Philip Code Academy. All Rights Reserved.</center>", unsafe_allow_html=True)

# ----------------------------#
# PROJECT OVERVIEW
# ----------------------------#
elif page == "📖 Project Overview":
    st.title("📊 Project Overview - SH17 Dataset")

    st.markdown("""
    The **SH17 Dataset** is a curated collection of annotated images developed for **object detection and classification tasks**.  
    It has been designed to train computer vision systems for **real-world, multi-class object identification**.

    #### 🧠 Dataset Summary:
    - **Name:** SH17 Dataset  
    - **Total Images:** 8,000+ labeled samples  
    - **Image Resolution:** Variable (640×640 optimized for YOLOv8)  
    - **Annotation Format:** Pascal VOC / YOLO text format  
    - **Use Case:** Real-time object detection model training  

    #### 📸 Dataset Composition:
    The dataset contains diverse categories of objects captured in various lighting and background conditions.  
    Each object is enclosed within a **bounding box**, labeled with its class name, and annotated for supervised model training.

    #### 🏷️ Classes Detected:
    The YOLO model trained on SH17 Dataset can detect the following 17 classes:

    - Person
    - Ear
    - Earmuffs
    - Face
    - Face-guard
    - Face-mask-medical
    - Foot
    - Tools
    - Glasses
    - Gloves
    - Helmet
    - Hands
    - Head
    - Medical-suit
    - Shoes
    - Safety-suit
    - Safety-vest

    These classes were carefully selected to represent **everyday objects relevant to urban, domestic, and research contexts.**

    ---
    ### 🧩 Model Overview
    The YOLO11n model was trained using the SH17 dataset for **70 epochs** with **early stopping** and **augmentation strategies** including:
    - Horizontal flips  
    - Random brightness adjustment  
    - Mosaic augmentation  
    - Scale normalization  

    The best model checkpoint was automatically saved based on the **highest mAP@0.5 metric**.

    ---
    Below is a sample detection result demonstrating bounding boxes generated by YOLO:
    """)

    st.image("sample.png",
             caption="Sample detection showing bounding boxes using YOLO", 
             use_container_width=True)

    st.image("sample (2).png",
             caption="Sample detection showing bounding boxes using YOLO", 
             use_container_width=True)

    st.image("sample (3).png",
             caption="Sample detection showing bounding boxes using YOLO", 
             use_container_width=True)
    
    st.markdown("<br><hr><center>© 2025 Philip Code Academy. All Rights Reserved.</center>", unsafe_allow_html=True)

# ----------------------------#
# TRY THE MODEL PAGE
# ----------------------------#
elif page == "🧩 Try the Model":
    st.title("🧠 Try the Object Detection Model")

    option = st.radio("Select Input Type:", ["Image", "Video", "Use Camera"])

    if option == "Image":
        image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if image_file:
            detect_objects(image_file, source_type="image")

    elif option == "Video":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file:
            detect_objects(video_file, source_type="video")

    elif option == "Use Camera":
        detect_objects(None, source_type="webcam")

    st.markdown("<br><hr><center>© 2025 Philip Code Academy. All Rights Reserved.</center>", unsafe_allow_html=True)

# ----------------------------#
# ABOUT PAGE
# ----------------------------#
elif page == "ℹ️ About":
    st.title("👥 About This Project")

    st.markdown("""
    The **YOLO Object Detection (SH17 Project)** was designed and implemented under the guidance of **Philip Code Academy**.  
    It showcases how deep learning, data labeling, and model deployment intersect to solve computer vision challenges.

    ---
    **Frameworks Used:**
    - Ultralytics YOLOv8  
    - PyTorch  
    - OpenCV  
    - Streamlit  

    **Core Objective:**  
    To build an end-to-end AI application — from dataset curation, training, to real-time deployment — in an accessible web format.

    ---
    For further training on Python, Data Analysis, and Machine Learning, visit **Philip Code Academy**’s official channels.
    """)

    st.markdown("<br><hr><center>© 2025 Philip Code Academy. All Rights Reserved.</center>", unsafe_allow_html=True)
