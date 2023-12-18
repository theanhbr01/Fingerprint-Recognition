import numpy as np
import cv2
from skimage import filters, morphology

def Normalize(image, M0, STD0, logging = False):
    """
    im_arr: image as array
    M0: desired mean
    VAR0: desired variance

    G: normalized image as array
    """

    row, col = image.shape

    if(logging):
        print("row col = ", row, col)

    # Mean and variance calculation
    M = (1 / (row ** 2)) * np.sum(image)
    VAR = (1 / (row ** 2)) * np.sum((image - M) ** 2)

    G = np.zeros(image.shape)
    for i in range(row):
        for j in range(col):

            if image[i, j] > M:
                G[i, j] = M0 + np.sqrt((STD0 * (image[i, j] - M) ** 2) / VAR)
            else:
                G[i, j] = M0 - np.sqrt((STD0 * (image[i, j] - M) ** 2) / VAR)

    return G

# get fingerprint region for crop
def Image_segmentation(image, crop_width, crop_height):
    CropWidth = crop_width
    CropHeight = crop_height
    EXPAND_WIDTH = 230
    EXPAND_HEIGHT = 230

    thresh = filters.threshold_otsu(image)

    bw = morphology.closing(image > thresh, morphology.square(3))

    cleared = bw.copy()

    img_width = image.shape[1]
    img_height = image.shape[0]

    crop_l = img_width
    crop_r = 0
    crop_t = img_height
    crop_b = 0
    for i in range(img_height):
        for j in range(img_width):
            if cleared[i, j] == False:
                if (crop_l > j):
                    crop_l = j
                if (crop_r < j):
                    crop_r = j
                if (crop_t > i):
                    crop_t = i
                if (crop_b < i):
                    crop_b = i

    if ((crop_r - crop_l) < CropWidth):
        diff = CropWidth - (crop_r - crop_l)
        if (crop_r + crop_l > CropWidth): # right
            if (img_width - crop_r > diff / 2):
                crop_r += diff / 2
                crop_l -= diff / 2
            else:
                crop_r = img_width - 1
                crop_l = crop_r - (CropWidth + 2)
        else: # left
            if (crop_l > diff / 2):
                crop_l -= diff / 2
                crop_r += diff / 2
            else:
                crop_l = 1
                crop_r = crop_l + (CropWidth + 2)
    if ((crop_b - crop_t) < CropHeight):
        diff = CropHeight - (crop_b - crop_t)
        if (crop_b + crop_t > CropHeight): # bottom
            if (img_height - crop_b > diff / 2):
                crop_b += diff / 2
                crop_t -= diff / 2
            else:
                crop_b = img_height - 1
                crop_t = crop_b - (CropHeight + 2)
        else: # top
            if (crop_t > diff / 2):
                crop_t -= diff / 2
                crop_b += diff / 2
            else:
                crop_t = 1
                crop_b = crop_t + (CropHeight + 2)

    # expand region for rotation
    crop_l = (crop_r + crop_l - CropWidth) / 2
    crop_r = crop_l + CropWidth
    crop_t = (crop_t + crop_b - CropHeight) / 2
    crop_b = crop_t + CropHeight
    crop_l = (int)(crop_l - ((EXPAND_WIDTH - CropWidth) / 2))
    crop_r = (int)(crop_r + ((EXPAND_WIDTH - CropWidth) / 2))
    crop_t = (int)(crop_t - ((EXPAND_HEIGHT - CropHeight) / 2))
    crop_b = (int)(crop_b + ((EXPAND_HEIGHT - CropHeight) / 2))

    # check expanded region
    diff = 0
    if (crop_l < 0):
        diff = 0 - crop_l
        crop_l = crop_l + diff
        crop_r = crop_r + diff
    if (crop_r >= img_width):
        diff = crop_r - (img_width - 1)
        crop_l = crop_l - diff
        crop_r = crop_r - diff

    diff = 0
    if (crop_t < 0):
        diff = 0 - crop_t
        crop_t = crop_t + diff
        crop_b = crop_b + diff
    if (crop_b >= img_height):
        diff = crop_b - (img_height - 1)
        crop_t = crop_t - diff
        crop_b = crop_b - diff
        
    cropped_array = [row[crop_l:crop_r] for row in image[crop_t:crop_b]]

    return cropped_array
