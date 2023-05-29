import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import numpy as np
from PIL import Image
from numpy import asarray

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray


# Read images

finger_name = glob.glob('D:/Nextcloud/Promotion/Projekt_Gummifinger/Bilder/Fingerbending/*.jpg')
image = Image.open(finger_name[105])
print(finger_name[105])
grayscale = rgb2gray(asarray(image))

# apply threshold. Use ~30/255 or 40/255 for black and silicone grippers and ~90/255 for silica grippers
#thresh = threshold_otsu(grayscale)
thresh = 90/255
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

plt.pcolormesh(bw)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 6000:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()