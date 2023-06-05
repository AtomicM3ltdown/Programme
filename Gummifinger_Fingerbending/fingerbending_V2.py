import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import numpy as np
import os
from PIL import Image
from numpy import asarray
from scipy.ndimage import zoom
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
import math
import csv
from datetime import date


def intercept(point1, slope1, point2, slope2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the x-coordinate of the intersection point
    x = (y2 - y1 + slope1 * x1 - slope2 * x2) / (slope1 - slope2)

    # Calculate the y-coordinate of the intersection point
    y = slope1 * (x - x1) + y1

    return (x, y)


def calculate_intersection(line1, line2):
    """
    Calculates the point of intersection between two lines.

    Parameters:
    line1 (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the first line.
    line2 (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the second line.

    Returns:
    tuple: (x, y) coordinates of the point of intersection.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate the slopes of the lines
    slope1 = (y2 - y1) / (x2 - x1)
    slope2 = (y4 - y3) / (x4 - x3)

    # Check if the lines are parallel (same slope)
    if slope1 == slope2:
        return None  # Lines are parallel, no intersection

    # Calculate the x-coordinate of the intersection point
    x = (slope1 * x1 - slope2 * x3 + y3 - y1) / (slope1 - slope2)

    # Calculate the y-coordinate of the intersection point
    y = slope1 * (x - x1) + y1

    return x, y


# Calculate the length of a line segment
def calculate_length(x1, y1, x2, y2):
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


# Calculate the angle between two lines
def cosine_rule(a, b, c):
    """
    Calculates the angle (in degrees) opposite to side c using the cosine rule.

    Parameters:
    a (float): Length of side a
    b (float): Length of side b
    c (float): Length of side c

    Returns:
    float: Angle (in degrees) opposite to side c
    """
    numerator = a ** 2 + b ** 2 - c ** 2
    denominator = 2 * a * b
    cos_angle = numerator / denominator

    # Check if cos_angle is within the valid range [-1, 1]
    if cos_angle < -1:
        cos_angle = -1
    elif cos_angle > 1:
        cos_angle = 1

    angle = math.degrees(math.acos(cos_angle))
    return angle


def write_to_csv(finger_name, finger_angle2, deflection_x, deflection_y):
    base_path = r'D:/Nextcloud/Promotion/Projekt_Gummifinger/01_Ergebnisse/03_Druck_Auslenkung'
    current_date = date.today().strftime("%Y-%m-%d")
    file_path = os.path.join(base_path, f'results_{current_date}.csv')

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Finger Name', 'Finger Angle2', 'Deflection X', 'Deflection Y'])
        writer.writerow([finger_name, finger_angle2, deflection_x, deflection_y])


# Read images

finger_names = glob.glob('D:/Nextcloud/Promotion/Projekt_Gummifinger/00_Bilder/Fingerbending_black/*.jpg')
for finger_name in finger_names:
    image = Image.open(finger_name)
    print(finger_name)

    grayscale = rgb2gray(asarray(image))

    # apply threshold. Use ~30/255 or 40/255 for black and silicone grippers and ~90/255 for silica grippers
    # thresh = threshold_otsu(grayscale)
    thresh = 30 / 255
    binary = grayscale > thresh

    # convert binary to the same shape as bw
    binary = binary.astype(int)

    # apply closing to binary image
    bw = np.invert(closing(binary, square(2)))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=grayscale, bg_label=0, bg_color=None, kind='overlay')

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(image_label_overlay)
    ax.set_axis_off()

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 5000:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            # Crop and resize the image based on the rectangle's position and size
            scale_factor = 1  # Adjust the scale factor as needed
            cropped_image = asarray(label_image)[int(minr):int(maxr), int(minc):int(maxc)]
            resized_image = zoom(cropped_image, scale_factor)
            ax.imshow(resized_image, extent=(minc, maxc, maxr, minr), alpha=0.5)

            fig, ax = plt.subplots(figsize=(3, 4))
            ax.imshow(resized_image)
            # plt.pcolormesh(mask)

            # Find the coordinates of the first pixel with a value higher than 100
            mask = cropped_image > 30
            mask_indices = np.argwhere(mask)
            if mask_indices.shape[0] > 0:
                # Sort the mask indices based on the sum of row and column in ascending order
                sorted_indices = mask_indices[np.argsort(mask_indices[:, 0] + mask_indices[:, 1])]

                # Get the furthest left and bottom pixel coordinates
                red_pixel = sorted_indices[0]

                # Calculate the coordinates in the original image
                red_pixel_orig = (
                    int(red_pixel[1] / scale_factor),
                    int(red_pixel[0] / scale_factor)
                )
                # Plot marker on the original image
                ax.plot(*red_pixel_orig, 'ro', markersize=10)
            if mask_indices.shape[0] > 0:
                # Sort the mask indices based on the column coordinate in ascending order
                sorted_indices = mask_indices[np.argsort(mask_indices[:, 1])]

                # Calculate the number of pixels in the bottom 3% of the mask
                num_pixels_bottom_3_percent = int(mask_indices.shape[0] * 0.3)
                # Calculate the threshold value for filtering
                threshold = mask.shape[0] - mask.shape[0] * 0.03

                # Filter out indices where [0] is lower than the threshold
                mask_indices = mask_indices[mask_indices[:, 0] >= threshold]

                # Get the furthest left pixel coordinates in the bottom 3% of the mask
                furthest_left_bottom_3_percent_pixel = sorted_indices[-num_pixels_bottom_3_percent:]

                # Calculate the coordinates in the original image
                blue_pixel = (
                    int(mask_indices[0, 1] / scale_factor),
                    int(mask_indices[0, 0] / scale_factor)
                )

                # Plot marker on the original image
                ax.plot(*blue_pixel, 'bo', markersize=10)
                last_pixel = mask_indices[-1]
                yellow_pixel = [1, 2]
                yellow_pixel[0] = last_pixel[1]
                yellow_pixel[1] = last_pixel[0]
                yellow_pixel = asarray(yellow_pixel)
                ax.plot(*yellow_pixel, 'yo', markersize=10)

                # Calculate the slope of the red line
                red_slope = (blue_pixel[1] - yellow_pixel[1]) / (
                        blue_pixel[0] - yellow_pixel[0])

                # Calculate the slope of the orthogonal line (negative reciprocal)
                ortho_slope = -1 / red_slope

                # Get a point on the orthogonal line by adding a small displacement to the blue point
                displacement = 100  # Adjust the displacement as needed
                orthogonal_point_x = blue_pixel[0] - displacement / ortho_slope
                orthogonal_point_y = blue_pixel[1] - (displacement)
                green_length = calculate_length(blue_pixel[0], blue_pixel[1], orthogonal_point_x, orthogonal_point_y)

                line1 = (orthogonal_point_x, orthogonal_point_y, blue_pixel[0], blue_pixel[1])
                line2 = (red_pixel_orig[0], red_pixel_orig[1], yellow_pixel[0], yellow_pixel[1])
                intersection = calculate_intersection(line1, line2)

                if intersection is not None:
                    print("Point of intersection:", intersection)
                    ax.plot(*intersection, 'go', markersize=10)
                    orthogonal_point_x = intersection[0]
                    orthogonal_point_y = intersection[1]
                    green_length = calculate_length(blue_pixel[0], blue_pixel[1], intersection[0], intersection[1])
                else:
                    print("Lines don't intercept, no intersection")
                # Plot the orthogonal line
                ax.plot([blue_pixel[0], orthogonal_point_x],
                        [blue_pixel[1], orthogonal_point_y], 'g--')
            '''
            # Draw lies (yellow-->blue) and (red-->blue)
            ax.plot([red_pixel_orig[0], blue_pixel[0]],
                    [red_pixel_orig[1], blue_pixel[1]],
                    'b--')
            ax.plot([blue_pixel[0], yellow_pixel[0]],
                    [blue_pixel[1], yellow_pixel[1]],
                    'r--')
            ax.plot([red_pixel_orig[0], yellow_pixel[0]],
                    [red_pixel_orig[1], yellow_pixel[1]],
                    'm--')
            '''
            blue_length = calculate_length(red_pixel_orig[0], red_pixel_orig[1], blue_pixel[0], blue_pixel[1])
            print("Blue line length:", blue_length)
            magenta_length = calculate_length(red_pixel_orig[0], red_pixel_orig[1], yellow_pixel[0], yellow_pixel[1])
            print("Magenta line length:", magenta_length)
            red_length = calculate_length(yellow_pixel[0], yellow_pixel[1], blue_pixel[0], blue_pixel[1])
            print("Red line length:", red_length)
            print("Green line length:", green_length)

            red_to_blue_angle = cosine_rule(blue_length, red_length, magenta_length)
            print("Angle between red and blue lines:", red_to_blue_angle)

            magenta_part_length = calculate_length(intersection[0], intersection[1], yellow_pixel[0], yellow_pixel[1])
            red_to_green_angle = cosine_rule(green_length, red_length, magenta_part_length)
            print("Angle between red and green lines:", red_to_green_angle)
            part_magenta_to_green_angle = cosine_rule(red_length, magenta_part_length, green_length)
            print("Angle between magenta and green lines:", part_magenta_to_green_angle)
            red_to_magenta_angle = cosine_rule(magenta_part_length, green_length, red_length)
            print("Angle between red and magenta lines:", red_to_magenta_angle)
            winkel = red_to_magenta_angle + part_magenta_to_green_angle + red_to_green_angle
            print(winkel)

            magenta_to_green_length = calculate_length(intersection[0], intersection[1], red_pixel_orig[0],
                                                       red_pixel_orig[1])
            finger_angle = cosine_rule(blue_length, green_length, magenta_to_green_length)
            print("Angle between red and blue lines:", finger_angle)
            finger_angle2 = cosine_rule(green_length, magenta_to_green_length, blue_length)
            print("Den hier: Angle between red and blue lines:", finger_angle2)
            finger_angle3 = cosine_rule(magenta_to_green_length, blue_length, green_length)
            print("Angle between red and blue lines:", finger_angle3)
            finger_winkel = finger_angle + finger_angle2 + finger_angle3
            print(finger_winkel)

            # Extrapolate the green line
            green_slope = (orthogonal_point_y - blue_pixel[1]) / (orthogonal_point_x - blue_pixel[0])

            # plot line from red point to intercept green line
            plt.axline(red_pixel_orig, slope=red_slope, linewidth=1, color='r')
            plt.axline(blue_pixel, slope=green_slope, linewidth=1, color='c')
            red_line = plt.axline(red_pixel_orig, slope=red_slope, linewidth=1, color='r')
            blue_line = plt.axline(blue_pixel, slope=green_slope, linewidth=1, color='c')

            plt.axline(red_pixel_orig, slope=green_slope, linewidth=1, color='m')
            plt.axline(blue_pixel, slope=red_slope, linewidth=1, color='b')
            black_line = plt.axline(blue_pixel, slope=red_slope, linewidth=1, color='g')
            green_line = plt.axline(red_pixel_orig, slope=red_slope, linewidth=1, color='r')
            intercept1 = intercept(red_pixel_orig, red_slope, blue_pixel, green_slope)
            intercept2 = intercept(blue_pixel, red_slope, red_pixel_orig, green_slope)

            normalize_factor = calculate_length(yellow_pixel[0], yellow_pixel[1], blue_pixel[0], blue_pixel[1]) / 2.2
            deflection_x = calculate_length(red_pixel_orig[0], red_pixel_orig[1], intercept1[0],
                                            intercept1[1]) / normalize_factor
            deflection_y = calculate_length(red_pixel_orig[0], red_pixel_orig[1], intercept2[0],
                                            intercept2[1]) / normalize_factor

            print(intercept1)
            print(intercept2)
            print(deflection_x)
            print(deflection_y)
            ax.plot(*intercept1, 'go', markersize=10)
            ax.plot(*intercept2, 'go', markersize=10)

            write_to_csv(finger_name, finger_angle2, deflection_x, deflection_y)
            processed_dir = 'Processed image'
            ax.set_axis_off()
            os.makedirs(processed_dir, exist_ok=True)
            filename = os.path.join(processed_dir, os.path.basename(finger_name) + '.png')
            plt.savefig(filename, bbox_inches='tight')

    '''
            # Given the above, calculate y intercept "b" in y=mx+b
            b = red_pixel_orig[1] - red_slope * red_pixel_orig[0]

            # Now draw two points around the input point
            pt1 = (red_pixel_orig[0] - 5, red_slope * (red_pixel_orig[0] - 5) + b)
            pt2 = (red_pixel_orig[0] + 5, red_slope * (red_pixel_orig[0] + 5) + b)

            # Draw two line segments around the input point
            plt.plot((pt1[0], red_pixel_orig[0]), (pt1[1], red_pixel_orig[1]), marker='o')
            plt.plot((red_pixel_orig[0], pt2[0]), (red_pixel_orig[1], pt2[1]), marker='o')
    '''
    plt.show()
