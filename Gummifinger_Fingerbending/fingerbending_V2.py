import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


def convert_to_black_and_white():
    global original_image, displayed_image, bw_image_data
    # Get the threshold value from the slider
    threshold_value = threshold_slider.get()

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert the image to black and white
    _, bw_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert the OpenCV image array to PIL Image
    pil_image = Image.fromarray(bw_image)

    # Resize the black and white image to fit within the GUI
    resized_image = pil_image.resize((400, 400))

    # Store the black and white image data
    bw_image_data = resized_image

    # Display the black and white image in the GUI
    displayed_image = ImageTk.PhotoImage(resized_image)
    image_label.config(image=displayed_image)


def open_image():
    # Open a file dialog to select the image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            global original_image, displayed_image, bw_image_data
            # Load the image using OpenCV
            original_image = cv2.imread(file_path)

            # Resize the image to fit within the GUI
            resized_image = cv2.resize(original_image, (400, 400))

            # Convert the OpenCV image array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

            # Display the image in the GUI
            displayed_image = ImageTk.PhotoImage(pil_image)
            image_label.config(image=displayed_image)

            # Clear the black and white image data
            bw_image_data = None

        except IOError:
            print("Unable to open image")


def update_image(*args):
    convert_to_black_and_white()


def find_contours():
    global original_image, displayed_image, bw_image_data
    if bw_image_data is not None:
        # Ensure that the image has a single channel (grayscale)
        if len(bw_image_data.split()) > 1:
            bw_image_data = bw_image_data.convert("L")

        # Convert the black and white image data to a NumPy array
        bw_image = np.array(bw_image_data)

        # Find contours in the black and white image
        contours, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw contours on
        image_with_contours = original_image.copy()

        # Draw the contours on the image
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

        # Convert the OpenCV image array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))

        # Resize the image to fit within the GUI
        resized_image = pil_image.resize((400, 400))

        # Display the image with contours in the GUI
        displayed_image = ImageTk.PhotoImage(resized_image)
        image_label.config(image=displayed_image)


def save_image():
    global displayed_image
    if displayed_image is not None:
        # Open a file dialog to select the save path
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            try:
                # Save the displayed image to the specified path
                displayed_image.save(file_path)
                print("Image saved successfully")
            except IOError:
                print("Unable to save image")


def place_dots():
    global original_image, displayed_image
    if original_image is not None:
        # Create a copy of the original image to draw dots on
        image_with_dots = original_image.copy()

        # Place the three dots on the image
        cv2.circle(image_with_dots, (50, 50), 5, (0, 0, 255), -1)  # Red dot
        cv2.circle(image_with_dots, (200, 200), 5, (255, 0, 0), -1)  # Blue dot
        cv2.circle(image_with_dots, (350, 350), 5, (128, 0, 128), -1)  # Maroon dot

        # Convert the OpenCV image array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image_with_dots, cv2.COLOR_BGR2RGB))

        # Resize the image to fit within the GUI
        resized_image = pil_image.resize((400, 400))

        # Display the image with dots in the GUI
        displayed_image = ImageTk.PhotoImage(resized_image)
        image_label.config(image=displayed_image)


# Create the main window
window = tk.Tk()
window.title("Image Processing App")

# Create the file menu
menu_bar = tk.Menu(window)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_image)
file_menu.add_command(label="Save", command=save_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menu_bar.add_cascade(label="File", menu=file_menu)
window.config(menu=menu_bar)

# Create the image label
image_label = tk.Label(window)
image_label.pack()

# Create the slider for thresholding
slider_frame = tk.Frame(window)
slider_frame.pack()

threshold_slider = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=400, command=update_image)
threshold_slider.set(128)  # Set the initial threshold value
threshold_slider.pack(side=tk.LEFT)

increment_button = tk.Button(slider_frame, text="+", command=lambda: threshold_slider.set(threshold_slider.get() + 1))
increment_button.pack(side=tk.LEFT, padx=5)

decrement_button = tk.Button(slider_frame, text="-", command=lambda: threshold_slider.set(threshold_slider.get() - 1))
decrement_button.pack(side=tk.LEFT, padx=5)

# Create the buttons
convert_button = tk.Button(window, text="Convert to Black and White", command=convert_to_black_and_white)
convert_button.pack()

find_contours_button = tk.Button(window, text="Find Contours", command=find_contours)
find_contours_button.pack()

# Create the save button
save_button = tk.Button(window, text="Save Image", command=save_image)
save_button.pack()

# Create the place dots button
place_dots_button = tk.Button(window, text="Place Dots", command=place_dots)
place_dots_button.pack()

# Initialize variables
original_image = None
displayed_image = None
bw_image_data = None

if __name__ == "__main__":
    # Start the GUI event loop
    # Start the main event loop
    window.mainloop()



