import os
from plantcv import plantcv as pcv
import numpy as np
import cv2
from typing import NoReturn
import matplotlib.pyplot as plt


RBG_CHANNELS = {
    "b": (255, 0, 0),
    "g": (0, 255, 0),
    "r": (0, 0, 255)
}
LABELED_COLORS = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Lightness": (128, 128, 128),
    "Hue": (0, 165, 255),
    "Saturation": (75, 0, 130),
    "Value": (255, 165, 0),
    "Green-Magenta": (255, 105, 180),
    "Blue-Yellow": (0, 255, 255),
}


def extract_name(path: str) -> str:
    """ Returns filename without extension """
    base = os.path.basename(os.path.normpath(path))
    stem, _ = os.path.splitext(base)
    return stem


class ImageTransformer:
    """ Class to handle the transformation process of images """

    def __init__(self, image, debug, writeimg, outdir, display):
        """ Initialize the transformation class with the given parameters """

        self.image = image
        self.debug = debug
        self.writeimg = writeimg
        self.outdir = outdir
        self.display = display

        try:
            pcv.params.debug_outdir = self.outdir
        except AttributeError:
            pass

        if self.writeimg:
            self.name = f'{self.outdir}/{extract_name(self.image)}'

        self.original_img = None

        self.blur_img = None

        self.final_mask = None
        self.ab_mask = None

        self.roi_objs = None

        self.mask_img = None
        self.detected_obj = None
        self.load_original()

    def load_original(self) -> NoReturn:
        """ Read the original image """
        img = pcv.readimage(filename=self.image)[0]
        if self.debug == "print":
            pcv.print_image(img, f'{self.name}_original.JPG')
        self.original_img = img

    def apply_gaussian_blur(self) -> NoReturn:
        """ Apply Gaussian blur to the image and saves the result """
        s = pcv.rgb2gray_hsv(self.original_img, channel="s")
        s_thresh = pcv.threshold.binary(s, threshold=60, object_type="light")
        s_gblur = pcv.gaussian_blur(s_thresh, (5, 5), 0, None)

        if self.debug == "print":
            if len(s_gblur.shape) == 2:
                pcv.print_image(s_gblur, f'{self.name}_gaussian_blur.JPG')
            else:
                gray_blur = cv2.cvtColor(s_gblur, cv2.COLOR_BGR2GRAY)
                pcv.print_image(gray_blur, f'{self.name}_gaussian_blur.JPG')

        self.blur_img = s_gblur

    def apply_mask(self) -> NoReturn:
        """ Apply a mask to the image using the Gaussian blur """
        if self.blur_img is None:
            self.apply_gaussian_blur()

        b = pcv.rgb2gray_lab(self.original_img, "b")
        b_thresh = pcv.threshold.binary(b, 200, "light")

        bs = pcv.logical_or(self.blur_img, b_thresh)
        masked = pcv.apply_mask(self.original_img, bs, "white")

        masked_a = pcv.rgb2gray_lab(masked, "a")
        masked_b = pcv.rgb2gray_lab(masked, "b")

        maskeda_thresh = pcv.threshold.binary(masked_a, 115, "dark")
        maskeda_thresh1 = pcv.threshold.binary(masked_a, 135, "light")
        maskedb_thresh = pcv.threshold.binary(masked_b, 128, "light")

        ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh)
        ab = pcv.logical_or(maskeda_thresh1, ab1)

        ab_fill = pcv.fill(ab, 200)

        final_masked_image = pcv.apply_mask(masked, ab_fill, "white")

        if self.debug == "print":
            pcv.print_image(final_masked_image, f'{self.name}_masked.JPG')

        self.final_mask = final_masked_image
        self.ab_mask = ab_fill

    def compute_roi(self) -> NoReturn:
        """ Define a region of interest (ROI) in the image """
        if self.final_mask is None:
            self.apply_mask()

        x, y, w, h = 0, 0, 250, 250
        roi1_np = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.int32
        )
        contours, _ = cv2.findContours(
            self.ab_mask.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        self.roi_objs = []

        for cnt in contours:
            for point in cnt:
                pt = (float(point[0][0]), float(point[0][1]))
                if cv2.pointPolygonTest(roi1_np, pt, False) >= 0:
                    self.roi_objs.append(cnt)
                    break

        if self.debug == "print":
            roi_masked = self.original_img.copy()

            roi_rect_mask = np.zeros(self.ab_mask.shape, dtype=np.uint8)
            cv2.rectangle(roi_rect_mask, (x, y), (x + w, y + h), 255, -1)

            roi_green_mask = cv2.bitwise_and(self.ab_mask, roi_rect_mask)
            green = np.array([0, 255, 0], dtype=np.uint8)

            roi_masked[roi_green_mask > 0] = green

            cv2.rectangle(roi_masked, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.imwrite(f'{self.name}_roi_masked.JPG', roi_masked)

        self.roi_masked = roi_masked

    def analyze_objects(self) -> NoReturn:
        """ Analyze the objects in the image """
        if self.roi_objs is None:
            self.compute_roi()

        analysis_image = self.original_img.copy()
        if self.roi_objs:
            obj = max(self.roi_objs, key=cv2.contourArea)
        else:
            obj = None
        self.mask_img = self.ab_mask

        analysis_image = pcv.analyze.size(self.original_img, self.ab_mask)

        if self.debug == "print":
            pcv.print_image(
                analysis_image,
                f'{self.name}_analysis_objects.JPG'
            )
        self.color_analysis = analysis_image

        analysis_image = self.original_img.copy()

        if obj is not None:
            cv2.drawContours(analysis_image, [obj], -1, (0, 255, 0), 5)
            x, y, w, h = cv2.boundingRect(obj)
            cv2.rectangle(
                analysis_image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            boundary_img1 = self.original_img.copy()
            cv2.line(boundary_img1, (0, 1680),
                     (self.original_img.shape[1], 1680), (0, 0, 255), 2)

            chans = cv2.split(self.original_img)
            hist_canvas = np.zeros((300, 256, 3), dtype="uint8")

            for (chan, color) in zip(chans, 'rgb'):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                cv2.normalize(
                    hist,
                    hist,
                    alpha=0,
                    beta=300,
                    norm_type=cv2.NORM_MINMAX)
                hist = hist.flatten()
                for x, y in enumerate(hist):
                    cv2.line(
                        hist_canvas,
                        (x, 300),
                        (x, 300 - int(y)),
                        RBG_CHANNELS[color],
                        1,
                    )
        self.detected_obj = obj

    def generate_pseudolandmarks(self) -> NoReturn:
        """ Generate pseudolandmarks for the image """
        if self.mask_img is None:
            self.analyze_objects()

        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            self.original_img,
            self.mask_img,
            pcv.params.sample_label
        )

        if self.debug == "print":
            annotated = self.original_img.copy()
            for pt in top:
                pt_arr = np.asarray(pt).flatten()
                cv2.circle(
                    annotated,
                    center=(int(pt_arr[0]), int(pt_arr[1])),
                    radius=3,
                    color=(255, 0, 0),
                    thickness=-1
                )
            for pt in center_v:
                pt_arr = np.asarray(pt).flatten()
                cv2.circle(
                    annotated,
                    center=(int(pt_arr[0]), int(pt_arr[1])),
                    radius=3,
                    color=(0, 69, 255),
                    thickness=-1
                )
            for pt in bottom:
                pt_arr = np.asarray(pt).flatten()
                cv2.circle(
                    annotated,
                    center=(int(pt_arr[0]), int(pt_arr[1])),
                    radius=3,
                    color=(255, 19, 240),
                    thickness=-1
                )
            pcv.print_image(annotated, f'{self.name}_pseudolandmarks.JPG')

        self.pseudolandmarks = annotated

    def generate_color_histogram(self) -> NoReturn:
        """ Generate a color histogram for the image with 9 channels """
        if self.mask_img is None:
            self.analyze_objects()

        b, g, r = cv2.split(self.original_img)
        lab = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)

        lab_L, lab_a, lab_b = cv2.split(lab)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)

        channel_list = [
            ("Blue", b, 256),
            ("Green", g, 256),
            ("Red", r, 256),
            ("Lightness", lab_L, 256),
            ("Hue", hsv_h, 180),
            ("Saturation", hsv_s, 256),
            ("Value", hsv_v, 256),
            ("Green-Magenta", lab_a, 256),
            ("Blue-Yellow", lab_b, 256),
        ]

        plt.figure(figsize=(10, 6))
        for label, img, bins in channel_list:
            hist = cv2.calcHist([img], [0], None, [bins], [0, bins])
            hist = cv2.normalize(
                hist,
                hist,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX
            ).flatten()
            plt.plot(
                hist,
                label=label,
                color=np.array(LABELED_COLORS[label]) / 255
            )

        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Normalized Frequency")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

    def plot_sub(self, axes, pos, img, title):
        """Helper to plot a BGR image on given subplot position."""
        ax = axes[pos]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)

    def display_results(self) -> NoReturn:
        """ Display the results of the transformations """
        _, axes = plt.subplots(2, 3, figsize=(10, 6))
        self.plot_sub(axes, (0, 0), self.original_img, "Original Image")
        self.plot_sub(axes, (0, 1), self.blur_img, "Gaussian Blur")
        self.plot_sub(axes, (0, 2), self.final_mask, "Final Mask")
        self.plot_sub(axes, (1, 0), self.roi_masked, "ROI Masked Image")
        self.plot_sub(axes, (1, 1), self.color_analysis, "Color Analysis")
        self.plot_sub(axes, (1, 2), self.pseudolandmarks, "Pseudolandmarks")
        plt.tight_layout()

        self.generate_color_histogram()
