import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from skimage import morphology

def image_processing(image):
    # konwersja na obraz szary
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # usunięcie pikseli o bardzo ciemnym odcieniu szarości
    img_gray[img_gray < 35] = 255

    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0) #usunięcie szumu

    # normalizacja wartości pikseli
    MIN, MAX = 48, 140
    norm = (img_gray - MIN) / (MAX - MIN)  # wynik to znormalizowany obraz o wartościach pikseli z przedziału 0-1
    norm = np.clip(norm, 0, 1) * 255  # ograniczenia wartości pikseli w znormalizowanym obrazie do przedziału 0-1,
                                      # mnozymy *255, aby uzyskać wartości pikseli z zakresu 0-255

    kernel_size = 5
    padded_image = np.pad(norm, (kernel_size // 2, kernel_size // 2))
    filtered_image = np.zeros_like(norm)
    for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
            filtered_image[i - kernel_size // 2, j - kernel_size // 2] = np.median(padded_image[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1])

    #ret, filtered_image = cv2.threshold(filtered_image, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    #filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    filtered_image = filtered_image.astype(np.uint8)
    #thresh2 = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    filtered_image = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #progowanie rozkład Gaussa
    #thresh4 = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #retOtsu, thresh5 = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    #filtered_image = cv2.erode(thresh3, kernel, iterations=1)
    #filtered_image = cv2.dilate(thresh3, kernel, iterations=1)
    #filtered_image = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel) #zamknięcie
    # filtered_image = morphology.remove_small_objects(filtered_image.astype(bool), min_size=100, connectivity=1).astype(
    #     np.uint8) * 255
    #filtered_image = cv2.morphologyEx(thresh3, cv2.MORPH_GRADIENT, kernel)
    #filtered_image = cv2.morphologyEx(thresh3, cv2.MORPH_BLACKHAT, kernel)

    #filtered_image = cv2.Laplacian(filtered_image, cv2.CV_64F)
    #filtered_image = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 1, ksize=5)



    length = len(filtered_image)
    for i in range(length):
        for j in range(len(filtered_image[i])):
            if filtered_image[i][j] == 255:
                filtered_image[i][j] = 0
            else:
                filtered_image[i][j] = 255

    return filtered_image