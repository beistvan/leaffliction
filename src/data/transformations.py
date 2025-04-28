""" Image transformation functions using real PlantCV """
from typing import Dict
import numpy as np
import cv2
from plantcv import plantcv as pcv


def get_transformations() -> Dict[str, callable]:
    return {
        "Original": transformation_original,
        "Gaussian_blur": transformation_gaussian_blur_bw,
        "Contrast_adjusted": transformation_contrast_adjusted,
        "Mask": transformation_mask_transparent,
        "Roi_objects": transformation_roi_objects,
        "Analyze_object": transformation_analyze_contour,
        "Pseudolandmarks": transformation_pseudolandmarks_real
    }


def binary_mask(image: np.ndarray) -> np.ndarray:
    image1 = pcv.threshold.dual_channels(
                rgb_img=image,
                x_channel='R',
                y_channel='B',
                points=[(1, 1), (image.shape[0]-1, image.shape[1]-1)],
                above=True
            )
    image2 = pcv.threshold.dual_channels(
                rgb_img=image,
                x_channel='R',
                y_channel='G',
                points=[(1, 1), (image.shape[0]-1, image.shape[1]-1)],
                above=True
            )
    image = pcv.logical_or(image1, image2)
    return image


def transformation_original(image: np.ndarray) -> np.ndarray:
    """ Return a copy of the original image """
    return image.copy()


def transformation_gaussian_blur_bw(image: np.ndarray) -> np.ndarray:
    """ Apply Gaussian blur and convert to black-and-white """
    return pcv.gaussian_blur(binary_mask(image), (11, 11), 0)


def transformation_contrast_adjusted(image: np.ndarray) -> np.ndarray:
    """ Stretch contrast using PlantCV's stretch function """
    gray = pcv.rgb2gray_lab(rgb_img=image, channel='l')  # LAB luminance
    # stretched = pcv.stretch(img=gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # Keep 3 channels for consistency


def transformation_mask_transparent(image: np.ndarray) -> np.ndarray:
    """ Apply HSV masking to isolate greenish areas and apply transparent background """
    mask = binary_mask(image)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    mask = cv2.bitwise_and(image, mask)  # Apply mask to image
    mask[np.where((mask == [0, 0, 0]).all(axis=2))] = [255, 255, 255]  # Set background to white
    return mask


def transformation_roi_objects(image: np.ndarray) -> np.ndarray:
    """ Draw bounding rectangles around detected leaf parts """
    mask_gray = cv2.cvtColor(transformation_mask_transparent(image), cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Blue box
    return output


def transformation_analyze_contour(image: np.ndarray) -> np.ndarray:
    """ Only draw largest contour, no text """
    mask_gray = cv2.cvtColor(transformation_mask_transparent(image), cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 2)  # Green contour
    return output


def transformation_pseudolandmarks_real(image: np.ndarray) -> np.ndarray:
    """ Detect pseudolandmarks (corners) properly """
    mask_gray = cv2.cvtColor(transformation_mask_transparent(image), cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

    corners = cv2.goodFeaturesToTrack(binary, maxCorners=50, qualityLevel=0.01, minDistance=10)

    output = image.copy()
    if corners is not None:
        corners = np.intp(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(output, (x, y), 3, (255, 0, 255), -1)  # Magenta color
    return output
