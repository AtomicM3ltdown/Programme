import cv2 as cv
from tkinter import Tk, Canvas, Button, filedialog, Image
from PIL import ImageTk
import sys

path = "D:/Nextcloud/Promotion/Projekt_Gummifinger/Bilder/Fingerbending/IMG-20230419-WA0010.jpg"

def create_contour_img(image_path, min_area, min_vertices):
    # Load the image
    image = cv.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply preprocessing (optional)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    _, threshold_image = cv.threshold(blurred_image, 100, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Process and filter contours
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

        # Filter based on area and number of vertices
        if area > min_area and len(approx) == min_vertices:
            cv.drawContours(image, [approx], -1, (0, 255, 0), 2)

    # Display or save the processed image
    cv.imshow("Processed Image", gray_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Convert OpenCV image to PIL Image
    processed_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    # Create a Tkinter window
    window = Tk()

    # Create a canvas to display the image
    canvas = Canvas(window, width=processed_image.width, height=processed_image.height)
    canvas.pack()

    # Convert PIL Image to Tkinter-compatible format
    image_tk = ImageTk.PhotoImage(processed_image)

    # Display the image on the canvas
    canvas.create_image(0, 0, anchor='nw', image=image_tk)

    # Start the Tkinter event loop
    window.mainloop()

# Usage example
path = filedialog.askopenfilename()
create_contour_img(path, 500, 4)