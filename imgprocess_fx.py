import cv2
import numpy as np

def add_gaussian_noise(image):
    noise = np.random.randint(0, 256, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 255

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def remove_noise_gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def remove_noise_median(image):
    return cv2.medianBlur(image, 5)

def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    hist_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return hist_img

def difference_of_gaussians(image, sigma1=1.0, sigma2=2.0): # بديل Canny
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, image)
    blur1 = cv2.GaussianBlur(gray_img, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(gray_img, (0, 0), sigma2)
    dog = cv2.subtract(blur1, blur2)

    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog.astype(np.uint8)


def non_max_suppression(magnitude, direction):  # المرحلة الثالثة في Canny
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)

    angle = direction
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                # Angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                # Angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                # Angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]

                if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                    Z[i,j] = magnitude[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def double_thresholding(nms, low_threshold, high_threshold): # المرحلة الرابعة والخامسة في Canny
    double_threshold = np.zeros_like(nms)
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))
    double_threshold[strong_i, strong_j] = strong
    double_threshold[weak_i, weak_j] = weak
    for i in range(1, double_threshold.shape[0]-1):
        for j in range(1, double_threshold.shape[1]-1):
            if double_threshold[i,j] == weak:
                if ((double_threshold[i+1, j-1] == strong) or (double_threshold[i+1, j] == strong) or (double_threshold[i+1, j+1] == strong)
                    or (double_threshold[i, j-1] == strong) or (double_threshold[i, j+1] == strong)
                    or (double_threshold[i-1, j-1] == strong) or (double_threshold[i-1, j] == strong) or (double_threshold[i-1, j+1] == strong)):
                    double_threshold[i, j] = strong
                else:
                    double_threshold[i, j] = 0
    return double_threshold.astype(np.uint8)

def canny_edge_detection(image, low_threshold=100, high_threshold=200, sigma=1.0):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #تحويل الصورة الى نسخة رمادية
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), sigma) #تطبيق Gaussian Blur لتقليل الضوضاء
    # حساب التدرجات باستخدام Sobel
    grad_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    nms = non_max_suppression(magnitude, direction)
    edges = double_thresholding(nms, low_threshold, high_threshold)
    return edges


