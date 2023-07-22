import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw

# Global variables
image_path = None
threshold_value = 127


def open_image():
    global image_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        update_image()


def update_image():
    if image_path:
        original_img = cv2.imread(image_path)
        processed_img, region_info = process_image(original_img)
        display_images(original_img, processed_img, region_info)


def process_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, int(threshold_value), 255, cv2.THRESH_BINARY)
    cleaned_img, region_info = clean_border_and_label_regions(binary_img)
    return cleaned_img, region_info


def clean_border_and_label_regions(binary_image):
    cleaned_image = binary_image.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    # Ignore background label (index 0)
    max_area_label = np.argmax(stats[1:, -1]) + 1
    cleaned_image[labels != max_area_label] = 0

    region_info = stats[max_area_label]
    return cleaned_image, region_info


def display_images(original_image, processed_image, region_info):
    original_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    original_img = Image.fromarray(original_img)
    processed_img = Image.fromarray(processed_img)

    original_img.thumbnail((400, 400))
    processed_img.thumbnail((400, 400))

    original_img = ImageTk.PhotoImage(original_img)
    processed_img = ImageTk.PhotoImage(processed_img)

    original_label.config(image=original_img)
    original_label.image = original_img

    processed_label.config(image=processed_img)
    processed_label.image = processed_img

    # Draw a rectangle around the largest region on the processed image
    x, y, w, h = region_info[0], region_info[1], region_info[2], region_info[3]
    processed_with_rectangle = processed_image.copy()
    cv2.rectangle(processed_with_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 2)

    processed_with_rectangle_img = cv2.cvtColor(processed_with_rectangle, cv2.COLOR_BGR2RGB)
    processed_with_rectangle_img_pil = Image.fromarray(processed_with_rectangle_img)

    draw = ImageDraw.Draw(processed_with_rectangle_img_pil)
    draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)

    processed_with_rectangle_img_pil.thumbnail((400, 400))
    processed_with_rectangle_img = ImageTk.PhotoImage(processed_with_rectangle_img_pil)

    processed_with_rectangle_label.config(image=processed_with_rectangle_img)
    processed_with_rectangle_label.image = processed_with_rectangle_img

    # Update the region_info_label
    region_info_label.config(text=f"Region Info: x={x}, y={y}, width={w}, height={h}")


def update_threshold(value):
    global threshold_value
    threshold_value = value
    update_image()
    threshold_label.config(text=f"Threshold: {value}")


# Create the main GUI window
root = tk.Tk()
root.title("Image Processing GUI")
root.geometry("1600x900")

# Create buttons and labels
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

image_frame = tk.Frame(root)
image_frame.pack(pady=10)

original_label = tk.Label(image_frame)
original_label.pack(side=tk.LEFT, padx=5, pady=5)

processed_label = tk.Label(image_frame)
processed_label.pack(side=tk.LEFT, padx=5, pady=5)

processed_with_rectangle_label = tk.Label(root)
processed_with_rectangle_label.pack(pady=10)

threshold_label = tk.Label(root, text="Threshold: 127")
threshold_label.pack()

region_info_label = tk.Label(root, text="Region Info:")
region_info_label.pack()

# Create a slider for adjusting the threshold
threshold_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=update_threshold)
threshold_slider.set(threshold_value)
threshold_slider.pack()

# Run the GUI event loop
root.mainloop()
