# 🛠️ Image Preprocessing Specialist Tool

This project provides a comprehensive web tool for image preprocessing, built with **Python**, **OpenCV**, and **NumPy**, and deployed using **Streamlit**. It allows users to visually analyze the effects of various image manipulation and computer vision algorithms.

Use images in "sample_images" folder to try out our tool!

## ✨ Features and Algorithms

The tool is equipped with methods across three main categories: Noise Generation, Denoising (Filtering), and Feature Detection (Edge Detection).

### 1. 📢 Noise Generation

| Method | Description | Key Principle |
| :--- | :--- | :--- |
| **Gaussian Noise** | Distributes pixel intensity values randomly according to a Gaussian distribution, simulating sensor noise. | Uses `np.random` to add general random intensity variation. |
| **Salt and Pepper Noise** | Randomly introduces isolated bright (white) and dark (black) pixels across the image. | Uses `np.random` to target specific pixels for extreme value assignment. |

### 2. 🧹 Denoising Filters

| Method | Description | Best for Filtering... | OpenCV Function |
| :--- | :--- | :--- | :--- |
| **Gaussian Blur** | Applies a weighted average across a pixel neighborhood to smooth the image. | **Gaussian Noise** | `cv2.GaussianBlur()`  |
| **Median Blur** | Replaces the center pixel value with the statistical median value in its neighborhood. | **Salt and Pepper Noise** | `cv2.medianBlur()`  |

### 3. 🖼️ Image Enhancement and Edge Detection

| Method | Description | Core Implementation Steps |
| :--- | :--- | :--- |
| **Histogram Equalization** | Redistributes the image's pixel intensity values to cover the full dynamic range, enhancing global contrast. | 1. Converts to YUV. 2. Equalizes the Luminance (Y) channel. 3. Converts back to BGR. |
| **Difference of Gaussians (DoG)** | A feature enhancement technique that finds details existing at a specific scale by comparing two blurred versions. | 1. **Grayscaling**. 2. Apply two Gaussian Blurs ($\sigma_1$ and $\sigma_2$). 3. **Subtract** $\text{Blur}_2$ from $\text{Blur}_1$. |
| **Canny Edge Detection** | A robust, multi-stage algorithm designed to find optimal, thin, continuous edges. | 1. **Noise Reduction** (Gaussian Blur). 2. **Gradient Calculation** (Sobel filters). 3. **Non-Max Suppression** (Thinning edges). 4. **Double Thresholding**. 5. **Hysteresis Tracking** (Connecting weak edges to strong ones).  |

---

## 💻 Technical Stack

* **Core Logic:** `imgprocess_fx.py` (Contains all custom function implementations).
* **Libraries:**
    * **OpenCV (`cv2`):** Primary library for image loading, filtering, and color space conversion.
    * **NumPy:** Used for efficient mathematical operations and array manipulation (image representation).
    * **Streamlit:** Used to create the interactive, user-friendly web interface.
