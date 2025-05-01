import os
from plantcv import plantcv as pcv
import numpy as np
import cv2
from typing import NoReturn


RBG_CHANNELS = {
    "b": (255, 0, 0),
    "g": (0, 255, 0),
    "r": (0, 0, 255)
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
        self.hierarchy_data = None
        self.kept_mask_data = None

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

        xor_img = pcv.logical_xor(maskeda_thresh, maskedb_thresh)
        xor_img_color = pcv.apply_mask(self.original_img, xor_img, "white")

        ab_fill = pcv.fill(ab, 200)

        final_masked_image = pcv.apply_mask(masked, ab_fill, "white")

        if self.debug == "print":
            pcv.print_image(final_masked_image, f'{self.name}_masked.JPG')
            pcv.print_image(xor_img_color, f'{self.name}_xor.JPG')

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
        roi_objects = []
        pcv.params.debug = self.debug

        for cnt in contours:
            for point in cnt:
                pt = (float(point[0][0]), float(point[0][1]))
                if cv2.pointPolygonTest(roi1_np, pt, False) >= 0:
                    roi_objects.append(cnt)
                    break

        if self.debug == "print":
            roi_masked = self.original_img.copy()

            roi_rect_mask = np.zeros(self.ab_mask.shape, dtype=np.uint8)
            cv2.rectangle(roi_rect_mask, (x, y), (x + w, y + h), 255, -1)

            roi_green_mask = cv2.bitwise_and(self.ab_mask, roi_rect_mask)
            green = np.array([0, 255, 0], dtype=np.uint8)

            roi_masked[roi_green_mask > 0] = green

            cv2.rectangle(roi_masked, (x, y), (x + w, y + h), (255, 0, 0), 4)
            roi_masked_path = f'{self.name}_roi_masked.JPG'
            cv2.imwrite(roi_masked_path, roi_masked)

        pcv.params.debug = None
        self.roi_objs = roi_objects
        self.hierarchy_data = None
        self.kept_mask_data = None

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

        pcv.params.debug = self.debug
        analysis_image = pcv.analyze.size(self.original_img, self.ab_mask)

        if self.debug == "print":
            pcv.print_image(
                analysis_image,
                f'{self.name}_analysis_objects.JPG'
            )
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
            shape_imgs = analysis_image

            if self.debug == "print":
                pcv.print_image(shape_imgs, f'{self.name}_shape_analysis.JPG')

            boundary_img1 = self.original_img.copy()
            cv2.line(boundary_img1, (0, 1680),
                     (self.original_img.shape[1], 1680), (0, 0, 255), 2)

            if self.debug == "print":
                pcv.print_image(
                    boundary_img1,
                    f'{self.name}_boundary_analysis.JPG'
                )

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
            if self.debug == "print":
                pcv.print_image(hist_canvas, f'{self.name}_color_analysis.JPG')
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

        self.top = top
        self.bottom = bottom
        self.center_v = center_v

    def generate_color_histogram(self) -> NoReturn:
        """ Generate a color histogram for the image with 9 channels """
        if self.mask_img is None:
            self.analyze_objects()

        canvas_width = 256
        canvas_height = 300
        canvas = np.ones((canvas_height, canvas_width, 3), dtype="uint8") * 255

        b, g, r = cv2.split(self.original_img)

        lab = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2LAB)
        lab_L, lab_a, lab_b = cv2.split(lab)

        hsv = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)

        channel_list = [
            ("Blue", b, 256, (255, 0, 0)),
            ("Green", g, 256, (0, 255, 0)),
            ("Red", r, 256, (0, 0, 255)),
            ("Lightness", lab_L, 256, (128, 128, 128)),
            ("Hue", hsv_h, 180, (0, 165, 255)),
            ("Saturation", hsv_s, 256, (75, 0, 130)),
            ("Value", hsv_v, 256, (255, 165, 0)),
            ("Green-Magenta", lab_a, 256, (255, 105, 180)),
            ("Blue-Yellow", lab_b, 256, (0, 255, 255)),
        ]

        for label, img, bins, color in channel_list:
            hist = cv2.calcHist([img], [0], None, [bins], [0, bins])
            cv2.normalize(
                hist,
                hist,
                alpha=0,
                beta=canvas_height,
                norm_type=cv2.NORM_MINMAX
            )
            hist = hist.flatten()
            scale = canvas_width / bins

            for i in range(1, bins):
                x1 = int((i - 1) * scale)
                x2 = int(i * scale)
                y1 = canvas_height - int(hist[i - 1])
                y2 = canvas_height - int(hist[i])
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

        if self.debug == "print":
            pcv.print_image(canvas, f'{self.name}_color_histogram.JPG')
