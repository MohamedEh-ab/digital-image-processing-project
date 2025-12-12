import streamlit as st
import cv2
import numpy as np
from imgprocess_fx import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    remove_noise_gaussian,
    remove_noise_median,
    histogram_equalization,
    difference_of_gaussians,
    canny_edge_detection
)

st.title("Preprocessing Specialist Tool")

uploaded_file = st.file_uploader("Choose an image for preprocessing", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Original Image")

    st.subheader("Preprocessing Options:")

    option = st.selectbox(
        "Select preprocessing operation:",
        [
            "Add Gaussian Noise",
            "Add Salt and Pepper Noise",
            "Remove Noise (Gaussian Blur)",
            "Remove Noise (Median Blur)",
            "Histogram Equalization",
            "Difference of Gaussians",
            "Canny Edge Detection"
        ]
    )

    if st.button("Apply Processing"):

        if option == "Add Gaussian Noise":
            processed = add_gaussian_noise(image)
        elif option == "Add Salt and Pepper Noise":
            processed = add_salt_and_pepper_noise(image)
        elif option == "Remove Noise (Gaussian Blur)":
            processed = remove_noise_gaussian(image)
        elif option == "Remove Noise (Median Blur)":
            processed = remove_noise_median(image)
        elif option == "Histogram Equalization":
            processed = histogram_equalization(image)
        elif option == "Difference of Gaussians":
            processed = difference_of_gaussians(image)
        elif option == "Canny Edge Detection":
            processed = canny_edge_detection(image)

        is_grayscale = len(processed.shape) == 2 or processed.shape[2] == 1

        if is_grayscale:
        # إذا كانت رمادية، نعرضها بدون تحديد القنوات
            st.image(processed, caption="Processed Image (Grayscale)")
        else:
        # إذا كانت ملونة (مثل إضافة الضوضاء أو Equalization)، نستخدم BGR
            st.image(processed, channels="BGR", caption="Processed Image")
