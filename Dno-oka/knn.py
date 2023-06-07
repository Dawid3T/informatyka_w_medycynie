from os import listdir
from random import randint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import math


def knn(image):
    image_green = image[:, :, 1]
    binary_output_image = np.copy(image_green)

    train_moments_hu = []
    train_pixel_color = []

    for image_file in listdir("images/input"):
        img = cv2.imread("images/input/" + image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_green = img[:, :, 1]

        binary = cv2.imread("images/binary/" + image_file.split(".")[0] + ".ah.ppm")
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

        train_data = ([], [])
        current_pixel_1_number = 0
        current_data_number = 0
        max_data_number = 200
        minimum_pixel_1_number = 50

        while current_data_number < max_data_number:
            x = randint(0, img_green.shape[0] - 1)
            y = randint(0, img_green.shape[1] - 1)

            pixel_color = 1 if binary[x][y] == 255 else 0
            if current_pixel_1_number < minimum_pixel_1_number:
                if pixel_color == 1:
                    current_pixel_1_number += 1
                else:
                    continue
            fragment_image = get_fragment_image(img_green, x, y)
            train_data[0].append(get_hu_moments(fragment_image))
            train_data[1].append(pixel_color)
            current_data_number += 1

        train_moments_hu += train_data[0]
        train_pixel_color += train_data[1]

    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(train_moments_hu, train_pixel_color)

    for x, row in enumerate(image_green):
        for y, _ in enumerate(row):
            fragment_image = get_fragment_image(image_green, x, y)
            hu_moments = get_hu_moments(fragment_image)
            binary_output_image[x][y] = 255 if knn_model.predict([hu_moments])[0] == 1 else 0


    return binary_output_image


def get_hu_moments(fragment_image):
    hu_moments = cv2.HuMoments(cv2.moments(fragment_image))
    normalized_hu_moments = []

    for moment in hu_moments:
        if moment[0] == 0:
            normalized_hu_moments.append(0)
        else:
            normalized_hu_moments.append(-1 * math.copysign(1.0, moment[0]) * math.log10(abs(moment[0])))

    return normalized_hu_moments


def get_fragment_image(image, x, y):
    fragment_size = 5
    image_with_pads = np.pad(image,
                             [(fragment_size // 2, fragment_size // 2), (fragment_size // 2, fragment_size // 2)],
                             'constant', constant_values=(0, 0))
    return image_with_pads[x - fragment_size // 2:x + fragment_size // 2, y - fragment_size // 2:y + fragment_size // 2]