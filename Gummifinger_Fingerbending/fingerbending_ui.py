import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from skimage.color import label2rgb
from skimage.measure import label

# Global variables
image_path = None
threshold_value = 15
min_area_threshold = 1000000  # Default minimum area threshold


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
    labeled_img = label(cleaned_image, background=0)
    region_info = find_regions_greater_than_threshold(labeled_img, min_area_threshold)
    # Convert the labeled image to a color image for visualization
    labeled_color_image = label2rgb(labeled_img, bg_label=0)

    # Create a PIL Image from the labeled color image array
    processed_with_rectangles = Image.fromarray((labeled_color_image * 255).astype(np.uint8))

    # Resize the image to a suitable size for display
    processed_with_rectangles.thumbnail((400, 400))

    # Convert back to PhotoImage
    processed_with_rectangles_img = ImageTk.PhotoImage(processed_with_rectangles)
    return cleaned_image, region_info


def find_regions_greater_than_threshold(labeled_image, area_threshold):
    regions = []
    _, counts = np.unique(labeled_image, return_counts=True)

    for label_val, count in enumerate(counts):
        if label_val == 0:  # Skip the background label (index 0)
            continue
        if count >= area_threshold:
            region_indices = np.where(labeled_image == label_val)
            x_min, y_min = np.min(region_indices[1]), np.min(region_indices[0])
            x_max, y_max = np.max(region_indices[1]), np.max(region_indices[0])
            width, height = x_max - x_min + 1, y_max - y_min + 1
            regions.append((x_min, y_min, width, height, count))

    return regions


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

    # Draw rectangles around the dark regions greater than the area threshold on the processed image
    processed_with_rectangles = Image.fromarray(processed_image)  # Create a PIL Image from the processed image array
    draw = ImageDraw.Draw(processed_with_rectangles)

    for region in region_info:
        x, y, width, height, area = region
        draw.rectangle([x, y, x + width, y + height], outline=1, width=5)  # Update outline color to (0, 255, 0)

    processed_with_rectangles.thumbnail((400, 400))  # Resize the image to a suitable size for display

    processed_with_rectangles_img = ImageTk.PhotoImage(processed_with_rectangles)  # Convert back to PhotoImage

    processed_with_rectangle_label.config(image=processed_with_rectangles_img)
    processed_with_rectangle_label.image = processed_with_rectangles_img

    # Update the region_info_label
    if len(region_info) > 0:
        region_info_str = "\n".join([f"x={x}, y={y}, width={width}, height={height}, area={area}" for x, y, width, height, area in region_info])
        region_info_label.config(text=f"Region Info:\n{region_info_str}")
    else:
        region_info_label.config(text="Region Info: No dark regions found.")

    # Update the resized_img_label
    if "resized_img" in globals():
        resized_img_label.config(image="")
        resized_img_label.image = ""


def update_threshold(value):
    global threshold_value
    threshold_value = value
    update_image()
    threshold_label.config(text=f"Threshold: {value}")


def update_min_area_threshold():
    global min_area_threshold
    try:
        min_area_threshold = int(min_area_entry.get())
        update_image()
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for the minimum area threshold.")


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

min_area_frame = tk.Frame(root)
min_area_frame.pack(pady=10)

min_area_label = tk.Label(min_area_frame, text="Minimum Area Threshold:")
min_area_label.pack(side=tk.LEFT)

min_area_entry = tk.Entry(min_area_frame, width=10)
min_area_entry.pack(side=tk.LEFT)

min_area_button = tk.Button(min_area_frame, text="Update", command=update_min_area_threshold)
min_area_button.pack(side=tk.LEFT)

region_info_label = tk.Label(root, text="Region Info:")
region_info_label.pack()

# Create a slider for adjusting the threshold
threshold_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, command=update_threshold)
threshold_slider.set(threshold_value)
threshold_slider.pack()

# Run the GUI event loop
root.mainloop()
