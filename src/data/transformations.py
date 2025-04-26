""" Image transformation functions for various image processing tasks """
from typing import Dict
import numpy as np
import cv2


def get_transformations() -> Dict[str, np.ndarray]:
    """ Returns a dictionary of all transformations """
    return {
        "Original": transformation_original,
        "Gaussian_blur": transformation_gaussian,
        "Contrast_adjusted": transformation_contrast,
        "Mask": transformation_mask,
        "Roi_objects": transformation_roi,
        "Analyze_object": transformation_analyze,
        "Pseudolandmarks": transformation_pseudolandmarks
    }


def transformation_original(image: np.ndarray) -> np.ndarray:
    """ Return a copy of the original image """
    return image.copy()


def transformation_gaussian(image: np.ndarray) -> np.ndarray:
    """ Apply Gaussian blur to the image """
    return cv2.GaussianBlur(image, (7, 7), 0)


def transformation_contrast(
    image: np.ndarray, alpha=1.5, beta=0
) -> np.ndarray:
    """ Adjust image contrast using scaling (alpha) and shifting (beta) """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def transformation_mask(image: np.ndarray) -> np.ndarray:
    """Apply HSV masking to isolate green-ish areas in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def transformation_roi(image: np.ndarray) -> np.ndarray:
    """ Draw bounding rectangles around masked regions (ROI detection) """
    mask = cv2.cvtColor(transformation_mask(image), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    image_roi = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image_roi


def transformation_analyze(image: np.ndarray) -> np.ndarray:
    """ Draw the largest contour and annotate its area and perimeter """
    mask = cv2.cvtColor(transformation_mask(image), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    image_analyze = image.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        cv2.drawContours(image_analyze, [largest], -1, (0, 255, 0), 2)
        cv2.putText(image_analyze, f"Area: {int(area)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image_analyze, f"Perimeter: {int(perimeter)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image_analyze


def transformation_pseudolandmarks(image: np.ndarray) -> np.ndarray:
    """ Detect pseudo-landmarks using Shi-Tomasi corner detection """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=10
    )
    image_landmarks = image.copy()
    if corners is not None:
        corners = np.intp(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image_landmarks, (x, y), 3, (255, 0, 255), -1)
    return image_landmarks
