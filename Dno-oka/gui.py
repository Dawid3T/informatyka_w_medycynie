from tkinter import *
from PIL import Image
from PIL import ImageTk
from image import *
import copy
from calculate_statistic import *
from knn import knn

# PARAMETRY:
WIDTH_WINDOW = 1500
HEIGHT_WINDOW = 700
WIDTH_IMAGE = 300
HEIGHT_IMAGE = 300

file_name = "im0082"

input_image = cv2.imread("images/input/" + file_name + ".ppm")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

binary_input_image = cv2.imread("images/binary/" + file_name + ".ah.ppm")
wizualizacja = cv2.imread("images/binary/" + file_name + ".ah.ppm")

binary_output_image = image_processing(input_image)
binary_output_image = knn(input_image)
# binary_output_image = image_processing(input_image)
accuracy, sensitivity, specificity, precision, f_measure, g_mean = calculate_statistics(binary_input_image, binary_output_image)

print("Trafność: " + str(accuracy))
print("Czułość: " + str(sensitivity))
print("Swoistość: " + str(specificity))
print("Precyzja: " + str(precision))
print("G-mean: " + str(g_mean))
print("F-measure: " + str(f_measure))

root = Tk()
root.title("Dno oka")
root.resizable(0, 0)

width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()

x_position = (width_screen / 2) - (WIDTH_WINDOW / 2)
y_position = (height_screen / 2) - (HEIGHT_WINDOW / 2)

root.geometry('%dx%d+%d+%d' % (WIDTH_WINDOW, HEIGHT_WINDOW, x_position, y_position))

f = Canvas(root, width=WIDTH_WINDOW, height=HEIGHT_WINDOW)
f.pack()

label_input_image = Label(root, text="Obraz wejściowy:")
label_input_image.pack()
label_input_image.place(x=165, y=50)
display_input_image = Image.fromarray(input_image)
resized_input_image = display_input_image.resize((WIDTH_IMAGE, HEIGHT_IMAGE), Image.NEAREST)
display_input_image = ImageTk.PhotoImage(resized_input_image)
f.create_image(212, 300, image=display_input_image)

label_binary_input_image = Label(root, text="Maska ekspercka:")
label_binary_input_image.pack()
label_binary_input_image.place(x=510, y=50)
display_binary_input_image = Image.fromarray(binary_input_image)
resized_binary_input_image = display_binary_input_image.resize((WIDTH_IMAGE, HEIGHT_IMAGE), Image.NEAREST)
display_binary_input_image = ImageTk.PhotoImage(resized_binary_input_image)
f.create_image(574, 300, image=display_binary_input_image)

label_binary_output_image = Label(root, text="Obraz wyjściowy:")
label_binary_output_image.pack()
label_binary_output_image.place(x=870, y=50)
display_binary_output_image = Image.fromarray(binary_output_image)
resized_binary_output_image = display_binary_output_image.resize((WIDTH_IMAGE, HEIGHT_IMAGE), Image.NEAREST)
display_binary_output_image = ImageTk.PhotoImage(resized_binary_output_image)
f.create_image(936, 300, image=display_binary_output_image)

for i in range(len(binary_input_image)):
    for j in range(len(binary_input_image[i])):
        if binary_input_image[i][j][0] == binary_output_image[i][j] and binary_input_image[i][j][0] == 255:
            wizualizacja[i][j] = [255,255,255]
        elif binary_input_image[i][j][0] == 0 and binary_output_image[i][j] == 255:
            wizualizacja[i][j] = [255, 0, 0]
        elif binary_input_image[i][j][0] == 255 and binary_output_image[i][j] == 0:
            wizualizacja[i][j] = [0, 255, 0]

label_wizualizacja = Label(root, text="Wizualizacja wyników:")
label_wizualizacja.pack()
label_wizualizacja.place(x=1255, y=50)
display_wizualizacja = Image.fromarray(wizualizacja)
resized_wizualizacja = display_wizualizacja.resize((WIDTH_IMAGE, HEIGHT_IMAGE), Image.NEAREST)
display_wizualizacja = ImageTk.PhotoImage(resized_wizualizacja)
f.create_image(1298, 300, image=display_wizualizacja)

cv2.imwrite("output_binary_image_" + file_name + ".png", binary_output_image)
# cv2.imwrite("output_image_" + file_name + ".png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

root.mainloop()


