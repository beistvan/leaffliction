"""
Intensity histogram plotting functions
- RGB, HSV, HLS, and opponent color spaces
- Normalized histograms for each channel
- Plotting with Matplotlib
"""
import cv2
import numpy as np


def get_rgb_channels(image: np.ndarray):
    """ Extract RGB channels from the image """
    return [
        (image[:, :, 0], 'r', 'Red'),
        (image[:, :, 1], 'g', 'Green'),
        (image[:, :, 2], 'b', 'Blue'),
    ]


def get_hsv_channels(image: np.ndarray):
    """ Extract HSV channels from the image """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return [
        (hsv[:, :, 0], 'orange', 'Hue'),
        (hsv[:, :, 1], 'purple', 'Saturation'),
        (hsv[:, :, 2], 'black', 'Value'),
    ]


def get_hls_channel(image: np.ndarray):
    """ Extract HLS channel from the image """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    return [(hls[:, :, 1], 'gray', 'Lightness')]


def get_opponent_channels(image: np.ndarray):
    """ Extract opponent channels from the image """
    by = compute_opponent_channel(
        image[:, :, 2], (image[:, :, 0], image[:, :, 1])
    )
    gm = compute_opponent_channel(
        image[:, :, 1], (image[:, :, 0], image[:, :, 2])
    )
    return [
        (by, 'y', 'Blue-Yellow'),
        (gm, 'm', 'Green-Magenta')
    ]


def compute_opponent_channel(primary: np.ndarray, others: tuple) -> np.ndarray:
    """ Compute an opponent channel like BY or GM """
    primary = primary.astype('float32')
    mean_others = sum(ch.astype('float32') for ch in others) / len(others)
    diff = primary - mean_others
    return np.clip(diff, 0, 255).astype('uint8')


def get_channels(image: np.ndarray) -> list:
    """ Get all channels (RGB, HSV, HLS, opponent) """
    channels = []
    channels += get_rgb_channels(image)
    channels += get_hsv_channels(image)
    channels += get_hls_channel(image)
    channels += get_opponent_channels(image)
    return channels


def plot_color_histogram(image: np.ndarray, ax) -> None:
    """ Plot normalized histograms for RGB, HSV, HLS, and opponent channels """
    channels = get_channels(image)

    max_val = 0
    for data, color, label in channels:
        hist = cv2.calcHist([data], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum() * 100
        max_val = max(max_val, hist.max())
        ax.plot(hist, color=color, label=label)

    ax.set_xlim([0, 256])
    ax.set_ylim([0, max_val * 1.05])
    ax.set_facecolor('whitesmoke')
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_title("Color Histogram")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(which='both', color='white', linestyle='-', linewidth=0.5)
    ax.legend(
        title="Color Channel",
        title_fontsize="small",
        fontsize="small",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )
