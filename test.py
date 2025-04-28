# !/usr/bin/python
import sys, traceback
import cv2
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt

def display(ind):
    image_ = f"images/Apple/Apple_scab/image ({ind}).JPG"
    img, path, filename = pcv.readimage(image_, mode='RGB')

    pcv.print_image(img, "original.png")
    vars = ['R', 'G', 'B', 'l', 'a', 'b', 'h', 's', 'v']
    fig, axes = plt.subplots(9, 9, figsize=(15, 15))
    fig.suptitle("9x9 Grid of Dual-Channel Thresholding", fontsize=16)

    for i, a in enumerate(vars):
        for j, b in enumerate(vars):
            thresh = pcv.threshold.dual_channels(
                rgb_img=img,
                x_channel=a,
                y_channel=b,
                points=[(1, 1), (img.shape[0]-1, img.shape[1]-1)],
                above=True
            )
            # pcv.print_image(thresh, f"output/thresh_{a}_{b}.png")
            ax = axes[i, j]
            thresh = pcv.gaussian_blur(thresh, (7, 7), 0)
            ax.imshow(thresh, cmap='gray')
            ax.set_title(f"{a}-{b}", fontsize=8)
            ax.axis('off')
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontsize=8)
    axes[0, 0].axis('off')
    # Adjust layout and show the grid
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    # Get options
    # for i in range(1, 20):
        display(123)


if __name__ == "__main__":
    main()
