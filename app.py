import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(page_title="Object Detection System")

st.title("Object Detection System")

# ---------------- LOAD MODEL ---------------- #

model = YOLO("best.pt")

# ---------------- SIDEBAR ---------------- #

option = st.sidebar.selectbox(
    "Select Mode",
    ["Image Detection", "Video Upload", "Webcam Capture"]
)

# ==================================================
# IMAGE DETECTION
# ==================================================

if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        temp_image = "temp.jpg"
        image.save(temp_image)

        results = model.predict(
            source=temp_image,
            conf=0.10
        )

        result_img = results[0].plot()

        st.image(
            result_img,
            caption="Detection Result",
            use_container_width=True
        )

        st.write("Detected Objects:")
        st.write(results[0].boxes)

# ==================================================
# VIDEO DETECTION
# ==================================================

elif option == "Video Upload":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.video(temp_video.name)

        if st.button("Detect Objects in Video"):

            with st.spinner("Processing video..."):

                model.predict(
                    source=temp_video.name,
                    save=True,
                    conf=0.10
                )

            st.success("Detection Completed")

            st.info(
                "Check output inside runs/detect/predict/"
            )

# ==================================================
# WEBCAM DETECTION
# ==================================================

elif option == "Webcam Capture":

    picture = st.camera_input("Take a Picture")

    if picture is not None:

        image = Image.open(picture)

        st.image(
            image,
            caption="Captured Image",
            use_container_width=True
        )

        image.save("camera.jpg")

        results = model.predict(
            source="camera.jpg",
            conf=0.10
        )

        result_img = results[0].plot()

        st.image(
            result_img,
            caption="Detection Result",
            use_container_width=True
        )

        st.write("Detected Objects:")
        st.write(results[0].boxes)