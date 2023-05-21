import cv2
from tkinter import Tk, Canvas, Button, filedialog, Image
from PIL import ImageTk


def create_contour_img(image_path, min_area, min_vertices):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing (optional)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process and filter contours
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Filter based on area and number of vertices
        if area > min_area and len(approx) == min_vertices:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    # Display or save the processed image
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Perform rubber gripper identification (implement your own logic here)
    create_contour_img(image, 500, 4)
    # Crop the image to remove unnecessary parts (implement your own logic here)

    # Convert the image from OpenCV format to Tkinter-compatible format
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(processed_image)
    processed_image = ImageTk.PhotoImage(processed_image)

    return processed_image



def open_image():
    # Open a file dialog to select the image file
    image_path = filedialog.askopenfilename()
    print(f'{image_path}',type(image_path))  # Print the image path for debugging
    # Process the image and display the result on the canvas
    processed_image = process_image(str(image_path))  # Convert the image_path object to a string
    canvas.create_image(0, 0, anchor='nw', image=processed_image)

# Create the main window
root = Tk()

# Create a canvas to display the image
canvas = Canvas(root, width=400, height=400)
canvas.pack()

# Create a button to open the image file
button = Button(root, text="Open Image", command=open_image)
button.pack()




if __name__ == "__main__":
    # Start the GUI event loop
    root.mainloop()


