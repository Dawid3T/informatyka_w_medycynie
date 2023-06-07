import numpy as np

def calculate_statistics(binary_input_image, binary_output_image):
    tp, tn, fp, fn = 0, 0, 0, 0 # inicjalizacja licznikow na 0
    # przechodzimy przez kazdy piksel w obrazie wejsciowym
    for i in range(len(binary_input_image)):
        for j in range(len(binary_input_image[0])):
            # pobieramy wartosc kazdego piksela z obrazu wej i wyj
            input_pixel = binary_input_image[i][j][0]
            output_pixel = binary_output_image[i][j]
            # jesli wartosci dwoch pikseli sa takie same
            if input_pixel == output_pixel:
                # sprawdzamy wartosc piksela 0- czarny piksel, 255- bialy piksel
                if input_pixel == 255:
                    tp += 1  # zgodnosc pozytywna
                else:
                    tn += 1  # zgodnosc negatywna
            else:  # wartosci pixeli sa rozne
                if input_pixel == 255:
                    fn += 1  # niedopasowanie negatywne
                else:
                    fp += 1 # niedopasowanie pozytywne

    accuracy = (tp+tn)/(tp+fp+fn+tn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    precision = tp/(tp+fp)

    # jakość klasyfikacji
    g_mean = np.sqrt(sensitivity*specificity)
    f_measure = (2 * precision * sensitivity) / (precision + sensitivity)


    return accuracy,sensitivity,specificity, precision, f_measure, g_mean
