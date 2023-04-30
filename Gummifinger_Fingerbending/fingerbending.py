import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
import traceback
import logging
from sklearn.cluster import KMeans

# set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_SHAPE = (1500, 1500)
IMAGE_HEIGHT = 1500
IMAGE_WIDTH = 1500
CROP_SIZE_LEFT = 200
CROP_SIZE_RIGHT = 200
CROP_SIZE_BOTTOM = 600
CROP_SIZE_TOP = 400
THRESHOLD = 30
DISTANCE_THRESHOLD = 100
DISTANCE_LINE_THRESHOLD = 50
HORIZONTALITY_THRESHHOLD = 50
CONTOUR_BOTTOM_PERCENTAGE = 0.9
TIP_CIRCLE_RADIUS = 20
BASE_CIRCLE_RADIUS = 20
ORIENTATION_CIRCLE_RADIUS = 20
TIP_COLOR = (0, 255, 0)
BASE_COLOR_LEFT = (0, 0, 255)
BASE_COLOR_RIGHT = (255, 0, 0)
OUTPUT_DIR = "./output/"


# ORIENTATION_COLOR = (255, 0, 0)


def load_image() -> tuple[np.ndarray, str, str]:
    """
    Open a file dialog to choose an image file and load the image.

    Returns:
        The loaded image as a NumPy array.
    """
    root = tk.Tk()
    root.withdraw()
    first_file_path = filedialog.askopenfilename(title='Choose the first picture file',
                                                 filetypes=[('Photo Files', '*.jpg')])
    dir_path, file_with_extension = os.path.split(first_file_path)

    # get the filename without extension
    filename_without_extension = os.path.splitext(file_with_extension)[0]
    image = cv2.imread(file_with_extension)
    return image, dir_path, first_file_path


def load_images() -> tuple[list[np.ndarray], str, list[str]]:
    """
    Open a file dialog to choose multiple image files and load the images.

    Returns:
        A tuple containing a list of loaded images as NumPy arrays, the directory path of the images, and a list of the file paths of the images.
    """
    root = tk.Tk()
    root.withdraw()
    files_path = filedialog.askopenfilenames(title='Choose picture files', filetypes=[('Photo Files', '*.jpg')])
    dir_path = os.path.dirname(files_path[0])
    images = []
    for file_path in files_path:
        filename = os.path.basename(file_path)
        filename_without_extension = os.path.splitext(filename)[0]
        image = cv2.imread(file_path)
        images.append(image)
    return images, dir_path, files_path


def crop_image(image: np.ndarray) -> np.ndarray:
    """
    Crop the input image to remove the borders.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The cropped image as a NumPy array.
    """
    height, width = image.shape[:2]
    cropped_image = image[CROP_SIZE_TOP:height - CROP_SIZE_BOTTOM, CROP_SIZE_LEFT:width - CROP_SIZE_RIGHT]
    return cropped_image


def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Resize the input image to a fixed size.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The resized image as a NumPy array.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Crop the image
    cropped_image = image[CROP_SIZE_TOP:height - CROP_SIZE_BOTTOM, CROP_SIZE_LEFT:width - CROP_SIZE_RIGHT]
    # cv2.imwrite(f"01_{img_name}_cropped.jpg", cropped_image)

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(f"01_{img_name}_cropped.jpg", thresh)
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour_idx = np.argmax([cv2.contourArea(c) for c in contours])
    contour = contours[largest_contour_idx]

    # Find the center point of the contour
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Crop the image around the center point with a width of 1000 pixels
    h, w = cropped_image.shape[:2]
    left = max(0, cx - (int(IMAGE_WIDTH / 2)))
    right = min(w, cx + (int(IMAGE_WIDTH / 2)))
    top = max(0, cy - (int(IMAGE_HEIGHT / 2)))
    bottom = min(h, cy + (int(IMAGE_HEIGHT / 2)))
    cropped_img = cropped_image[top:bottom, left:right]
    resized_img = cv2.resize(cropped_img, (IMAGE_HEIGHT, int(IMAGE_WIDTH * h / w)))
    # cv2.imwrite(f"01_{img_name}_cropped2.jpg", cropped_img)
    return resized_img


def threshold_image(image: np.ndarray) -> np.ndarray:
    """
    Threshold the input image to create a binary image.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The thresholded image as a NumPy array.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return thresh


def find_contours(image: np.ndarray) -> np.ndarray:
    """
    Find contours in the input binary image.

    Args:
        image: The input binary image as a NumPy array.

    Returns:
        The contours as a NumPy array.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_largest_contour(contours: np.ndarray) -> np.ndarray:
    """
    Find the contour with the largest area.

    Args:
        contours: The input contours as a NumPy array.

    Returns:
        The largest contour as a NumPy array.
    """
    if len(contours) == 0:
        return np.array([])

    # Find the contour with the largest area
    largest_contour_idx = np.argmax([cv2.contourArea(c) for c in contours])
    largest_contour = contours[largest_contour_idx]

    return largest_contour


def find_tip_alt(contour: np.ndarray) -> np.ndarray:
    """
    Find the tip of the finger.

    Args:
        contour: The input contour as a NumPy array.

    Returns:
        The tip of the finger as a NumPy array.
    """
    hull = cv2.convexHull(contour, returnPoints=False)
    # Find the defects in the convex hull
    defects = cv2.convexityDefects(contour, hull)
    # Find the indices of the defects in the convex hull
    defect_indices = defects[:, 0][:, 0]
    hull_indices = cv2.convexHull(contour, returnPoints=True)[:, 0][:, 0]
    defect_hull_indices = np.isin(hull_indices, defect_indices)

    # Find the indices of the defects that are adjacent in the convex hull
    adjacent_defect_indices = np.where(np.roll(defect_hull_indices, -1) & defect_hull_indices)[0]

    # Calculate the angles between adjacent defects
    angles = []
    for i in adjacent_defect_indices:
        p0 = contour[defect_indices[i - 1]][0]
        p1 = contour[defect_indices[i]][0]
        p2 = contour[defect_indices[i + 1]][0]
        v1 = np.array([p0[0] - p1[0], p0[1] - p1[1]])
        v2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cos_angle)
        angles.append(angle)

    # Find the index of the adjacent defects with the largest angle
    if len(angles) > 0:
        max_angle_idx = adjacent_defect_indices[np.argmax(angles)]
        tip_idx = defect_indices[max_angle_idx]
        tip = tuple(contour[tip_idx][0])
        return tip
    else:
        return None


def find_tip(contour: np.ndarray) -> np.ndarray:
    """
    Find the tip of the finger.

    Args:
        contour: The input contour as a NumPy array.

    Returns:
        The tip of the finger as a NumPy array.
    """
    # Find the convex hull of the contour
    hull = cv2.convexHull(contour, returnPoints=False)

    # Find the defects in the convex hull
    defects = cv2.convexityDefects(contour, hull)

    # Find the leftmost point of the contour
    tip = tuple(contour[contour[:, :, 0].argmin()][0])
    return tip


def find_base(contour: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the base of the finger.

    Args:
        contour: The input contour as a NumPy array.
        tip: The tip of the finger as a NumPy array.

    Returns:
        The left and right base points of the finger as NumPy arrays.
    """
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    # Get the number of points in the contour
    num_contour_points = contour.shape[0]

    # Find the number of points in the bottom x% of the contour
    num_bottom_points = int(num_contour_points * CONTOUR_BOTTOM_PERCENTAGE)

    # Get the bottom points of the contour
    bottom_points = contour[-num_bottom_points:, 0, :]

    # Find the leftmost and rightmost points among the bottom points
    leftmost_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
    rightmost_bottom = tuple(bottom_points[bottom_points[:, 0].argmax()])
    return leftmost_bottom, rightmost_bottom


def find_base_2(contour: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the base of the finger.

    Args:
        contour: The input contour as a NumPy array.
        tip: The tip of the finger as a NumPy array.

    Returns:
        The left and right base points of the finger as NumPy arrays.
    """
    # Get the number of points in the contour
    num_contour_points = contour.shape[0]

    # Find the number of points in the bottom x% of the contour
    num_bottom_points = int(num_contour_points * CONTOUR_BOTTOM_PERCENTAGE)

    # Get the bottom points of the contour
    bottom_points = contour[-num_bottom_points:, 0, :]

    # Get the x-coordinates of the bottom points
    bottom_xs = bottom_points[:, 0]

    # Compute the mean x-coordinate of the bottom points
    mean_bottom_x = np.mean(bottom_xs)

    # Find the leftmost and rightmost points among the bottom points that are closest to the mean x-coordinate
    leftmost_bottom = bottom_points[np.abs(bottom_xs - mean_bottom_x).argmin()]
    rightmost_bottom = bottom_points[np.abs(bottom_xs - mean_bottom_x).argmax()]

    return leftmost_bottom.astype(int), rightmost_bottom.astype(int)


def find_base_3(contour: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the base of the finger.

    Args:
        contour: The input contour as a NumPy array.
        tip: The tip of the finger as a NumPy array.

    Returns:
        The left and right base points of the finger as NumPy arrays.
    """
    # Find the bottom-most point of the contour
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    # Find the left and right base points
    leftmost_bottom = bottommost
    rightmost_bottom = bottommost
    for i in range(1, contour.shape[0]):
        left = contour[bottommost[1] - i, 0, :]
        right = contour[bottommost[1] - i, 0, :]
        if bottommost[0] - left[0] > DISTANCE_THRESHOLD:
            leftmost_bottom = tuple(left)
        if right[0] - bottommost[0] > DISTANCE_THRESHOLD:
            rightmost_bottom = tuple(right)
            break

    return leftmost_bottom, rightmost_bottom


def find_base_4(contour: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the longest almost straight line at the bottom of the object and mark the left and right points.

    Args:
        contour: The input contour as a NumPy array.

    Returns:
        The left and right points of the longest almost straight line at the bottom of the object as NumPy arrays.
    """
    # Get the bottommost point of the contour
    bottommost_point = tuple(contour[contour[:, :, 1].argmax()][0])

    # Find the points that are close to the bottommost point
    bottom_points = contour[contour[:, :, 1] >= bottommost_point[1] - 5]

    # Compute the horizontal distances between adjacent points
    distances = np.diff(bottom_points[:, 0])

    # Find the indices of the breaks in the horizontal distances
    breaks = np.where(distances > DISTANCE_LINE_THRESHOLD)[0] + 1

    # Split the bottom points into segments based on the breaks
    segments = np.split(bottom_points, breaks)

    # Keep only the segments that are almost horizontal
    almost_horizontal_segments = [segment for segment in segments if
                                  abs(segment[0][1] - segment[-1][1]) <= HORIZONTALITY_THRESHHOLD]

    # Find the longest almost horizontal segment
    longest_segment = max(almost_horizontal_segments, key=lambda s: s.shape[0])

    # Find the leftmost and rightmost points of the longest almost horizontal segment
    leftmost_point = longest_segment[longest_segment[:, 0].argmin()]
    rightmost_point = longest_segment[longest_segment[:, 0].argmax()]

    # Adjust for slight tilt
    if abs(leftmost_point[1] - rightmost_point[1]) > 5:
        m, b = np.polyfit(longest_segment[:, 0, 0], longest_segment[:, 0, 1], deg=1)
        leftmost_point[1] = int(m * leftmost_point[0] + b)
        rightmost_point[1] = int(m * rightmost_point[0] + b)

    return leftmost_point, rightmost_point


def find_base_5(contour: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find two points on an almost straight line in the bottom quarter of the contour, allowing for a slight angle of up to
    10 degrees.

    Args:
        contour: The input contour as a NumPy array.
        tip: The tip of the object as a NumPy array.

    Returns:
        The left and right points of the almost straight line at the bottom of the object as NumPy arrays.
    """
    # Get the bottom quarter of the contour
    bottom = contour[contour[:, :, 1] >= (contour[:, :, 1].max() * 3 / 4)]

    # Compute the distances between adjacent points
    distances = np.diff(bottom[:, 0])

    # Find the indices of the breaks in the distances
    breaks = np.where(distances > 10)[0] + 1

    # Split the bottom points into segments based on the breaks
    segments = np.split(bottom, breaks)

    # Keep only the segments that are almost horizontal
    almost_horizontal_segments = [segment for segment in segments if abs(segment[0][1] - segment[-1][1]) <= 5]

    # Find the longest almost horizontal segment
    longest_segment = max(almost_horizontal_segments, key=lambda s: s.shape[0])

    # Find the leftmost and rightmost points of the longest almost horizontal segment
    leftmost_point = longest_segment[longest_segment[:, 0].argmin()]
    rightmost_point = longest_segment[longest_segment[:, 0].argmax()]

    return leftmost_point, rightmost_point


def find_base_alternative(contour: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the base of the finger.

    Args:
        contour: The input contour as a NumPy array.
        tip: The tip of the finger as a NumPy array.

    Returns:
        The left and right base points of the finger as NumPy arrays.
    """
    # Get the number of points in the contour
    num_contour_points = contour.shape[0]

    # Find the number of points in the bottom x% of the contour
    num_bottom_points = int(num_contour_points * CONTOUR_BOTTOM_PERCENTAGE)

    # Get the bottom points of the contour
    bottom_points = contour[-num_bottom_points:, 0, :]

    # Cluster the bottom points using K-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(bottom_points)
    center_1, center_2 = kmeans.cluster_centers_

    # Assign the left and right base points based on their x-coordinates
    if center_1[0] < center_2[0]:
        leftmost_bottom = center_1
        rightmost_bottom = center_2
    else:
        leftmost_bottom = center_2
        rightmost_bottom = center_1

    return leftmost_bottom.astype(int), rightmost_bottom.astype(int)


def find_orientation(contour: np.ndarray, tip: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Find the orientation of the finger.

    Args:
        contour: The input contour as a NumPy array.
        tip: The tip of the finger as a NumPy array.
        base: The base of the finger as a NumPy array.

    Returns:
        The orientation of the finger as a NumPy array.
    """
    # Get the number of points in the contour
    num_contour_points = contour.shape[0]
    # Find the number of points in the bottom x% of the contour
    num_bottom_points = int(num_contour_points * CONTOUR_BOTTOM_PERCENTAGE)
    # Get the bottom points of the contour
    bottom_points = contour[-num_bottom_points:, 0, :]

    # Find the rightmost point among the bottom points
    orientation = tuple(bottom_points[bottom_points[:, 0].argmax()])
    return orientation


def draw_circles(image: np.ndarray, tip: np.ndarray, left_base: np.ndarray, right_base: np.ndarray,
                 largest_contour: np.ndarray) -> np.ndarray:
    """
    Draw circles on the input image at the tip, left base, and right base points.

    Args:
        image: The input image as a NumPy array.
        tip: The tip of the finger as a NumPy array.
        left_base: The left base of the finger as a NumPy array.
        right_base: The right base of the finger as a NumPy array.

    Returns:
        The input image with the circles drawn on it.
    """
    # Draw the tip circle
    cv2.circle(image, tuple(tip), TIP_CIRCLE_RADIUS, TIP_COLOR, -1)

    # Draw the left base circle
    cv2.circle(image, tuple(left_base), BASE_CIRCLE_RADIUS, BASE_COLOR_LEFT, -1)

    # Draw the right base circle
    cv2.circle(image, tuple(right_base), BASE_CIRCLE_RADIUS, BASE_COLOR_RIGHT, -1)
    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)

    return image


def save_edited_image(image: np.ndarray, filename: str, output_dir: str) -> str:
    """
    Save the edited image to a new directory.

    Args:
        image: The edited image as a NumPy array.
        filename: The filename of the original image.
        output_dir: The directory to save the edited image.

    Returns:
        The filename of the saved image.
    """
    dir_path, file_with_extension = os.path.split(filename)
    # get the filename without extension
    filename_without_extension = os.path.splitext(file_with_extension)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_filename = output_dir + "edited_" + file_with_extension
    # print(new_filename)
    cv2.imwrite(new_filename, image)
    return new_filename


images, dir_path, file_paths = load_images()
for i in range(len(images)):
    image = images[i]
    file_path = file_paths[i]
    logging.debug(image, file_path)
    try:
        resized_image = resize_image(image)
        thresh = threshold_image(resized_image)
        contours = find_contours(thresh)
        largest_contour = find_largest_contour(contours)
        tip = find_tip(largest_contour)
        left_base, right_base = find_base(largest_contour, tip)
        orientation = find_orientation(largest_contour, tip, right_base)
        # print(f"{resized_image}, {tip}, {tuple(left_base)}, {tuple(right_base)}, {largest_contour}")
        draw_circles(resized_image, tip, tuple(left_base), tuple(right_base), largest_contour)
        logging.debug(file_path, OUTPUT_DIR)

        # can be saved: resized_image, thresh
        save_edited_image(resized_image, file_path, OUTPUT_DIR)
        # logging.debug(cv2.imshow('finger', resized_image))
        # logging.debug(cv2.waitKey(0))
        # logging.debug(cv2.destroyAllWindows())
    except Exception as e:
        print("Error:", repr(e))
        traceback.print_exc()
        print(file_path)