import os
from plantcv import plantcv as pcv
import numpy as np
import cv2


# temporary
def extract_name(path: str) -> str:
    """ Returns filename without extension """
    base = os.path.basename(os.path.normpath(path))
    stem, _ = os.path.splitext(base)
    return stem


class ImageTransformer:
    """
    Class to handle the transformation process of images.
    """

    def __init__(self, parameters):
        """
        Initialize the transformation class with the given parameters.
        """
        self.parameters = parameters

        try:
            pcv.params.debug_outdir = self.parameters.outdir
        except AttributeError:
            pass

        if self.parameters.writeimg:
            self.name_save = f'{self.parameters.outdir}/{extract_name(self.parameters.image)}'

        self.original_img = None

        self.blur_img = None

        self.final_mask = None
        self.ab_mask = None

        self.roi_objs = None
        self.hierarchy_data = None
        self.kept_mask_data = None

        self.mask_img = None
        self.detected_obj = None

    def load_original(self):
        """
        Read the original image and save it to the class.
        """
        img = pcv.readimage(filename=self.parameters.image)[0]
        if self.parameters.debug == "print":
            pcv.print_image(
                img,
                filename=self.name_save + "_original.JPG",
            )
        self.original_img = img
        return img

    def apply_gaussian_blur(self):
        """
        Apply Gaussian blur to the image and save it to the class.
        """
        if self.original_img is None:
            self.load_original()
        s = pcv.rgb2gray_hsv(self.original_img, channel="s")
        s_thresh = pcv.threshold.binary(s, threshold=60, object_type="light")
        s_gblur = pcv.gaussian_blur(
            s_thresh, ksize=(
                5, 5), sigma_x=0, sigma_y=None)
        if self.parameters.debug == "print":
            if len(s_gblur.shape) == 2:
                pcv.print_image(
                    s_gblur,
                    filename=self.name_save + "_gaussian_blur.JPG",
                )
            else:
                gray_blur = cv2.cvtColor(s_gblur, cv2.COLOR_BGR2GRAY)
                pcv.print_image(
                    gray_blur,
                    filename=self.name_save + "_gaussian_blur.JPG",
                )
        self.blur_img = s_gblur
        return s_gblur

    def apply_mask(self):
        """
        Apply a mask to the image using the Gaussian blur
        and the b channel of the LAB color space.
        """
        if self.blur_img is None:
            self.apply_gaussian_blur()

        b = pcv.rgb2gray_lab(rgb_img=self.original_img, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=200, object_type="light")

        bs = pcv.logical_or(bin_img1=self.blur_img, bin_img2=b_thresh)
        masked = pcv.apply_mask(
            img=self.original_img,
            mask=bs,
            mask_color="white")

        masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")
        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, object_type="dark")
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, object_type="light")
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, object_type="light")

        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        xor_img = pcv.logical_xor(
            bin_img1=maskeda_thresh,
            bin_img2=maskedb_thresh)
        xor_img_color = pcv.apply_mask(
            img=self.original_img,
            mask=xor_img,
            mask_color="white")

        ab_fill = pcv.fill(bin_img=ab, size=200)

        final_masked_image = pcv.apply_mask(
            img=masked, mask=ab_fill, mask_color="white")

        if self.parameters.debug == "print":
            pcv.print_image(
                final_masked_image,
                filename=self.name_save +
                "_masked.JPG")
            pcv.print_image(
                xor_img_color,
                filename=self.name_save +
                "_xor.JPG")

        self.final_mask = final_masked_image
        self.ab_mask = ab_fill
        return final_masked_image

    def compute_roi(self):
        """
        Define a region of interest (ROI) in the image.
        """
        if self.final_mask is None:
            self.apply_mask()
        x, y, w, h = 0, 0, 250, 250
        roi1_np = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        contours, hierarchy = cv2.findContours(
            self.ab_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        id_objects = contours
        pcv.params.debug = self.parameters.debug
        roi_objects = []
        for cnt in id_objects:
            for point in cnt:
                pt = (float(point[0][0]), float(point[0][1]))
                if cv2.pointPolygonTest(roi1_np, pt, False) >= 0:
                    roi_objects.append(cnt)
                    break
        obj_area = sum([cv2.contourArea(c) for c in roi_objects])
        if self.parameters.debug == "print":
            file_rename = (
                f"{self.parameters.outdir}/"
                f"{pcv.params.device - 2}_obj_on_img.png"
            )
            file_delete = (
                f"{self.parameters.outdir}/"
                f"{pcv.params.device - 1}_roi_mask.png"
            )
            if os.path.exists(file_delete):
                os.remove(file_delete)
            if os.path.exists(file_rename):
                os.rename(file_rename, self.name_save + "_roi_mask.JPG")

            roi_masked = self.original_img.copy()

            roi_rect_mask = np.zeros(self.ab_mask.shape, dtype=np.uint8)
            cv2.rectangle(roi_rect_mask, (x, y), (x + w, y + h), 255, -1)

            roi_green_mask = cv2.bitwise_and(self.ab_mask, roi_rect_mask)
            green = np.array([0, 255, 0], dtype=np.uint8)

            roi_masked[roi_green_mask > 0] = green

            cv2.rectangle(roi_masked, (x, y), (x + w, y + h), (255, 0, 0), 4)
            roi_masked_path = self.name_save + "_roi_masked.JPG"
            cv2.imwrite(roi_masked_path, roi_masked)
        pcv.params.debug = None
        self.roi_objs = roi_objects
        self.hierarchy_data = None
        self.kept_mask_data = None
        return roi_objects, None, None, obj_area

    def analyze_objects(self):
        """
        Analyze the objects in the image.
        """
        if self.roi_objs is None:
            self.compute_roi()
        analysis_image = self.original_img.copy()
        if self.roi_objs:
            obj = max(self.roi_objs, key=cv2.contourArea)
        else:
            obj = None
        mask = self.ab_mask
        pcv.params.debug = self.parameters.debug
        analysis_image = pcv.analyze.size(
            img=self.original_img, labeled_mask=mask)
        if self.parameters.debug == "print":
            pcv.print_image(
                analysis_image,
                filename=self.name_save + "_analysis_objects.JPG",
            )
        analysis_image = self.original_img.copy()
        if obj is not None:
            cv2.drawContours(
                analysis_image, [obj], -1, (0, 255, 0), thickness=5)
            x, y, w, h = cv2.boundingRect(obj)
            cv2.rectangle(analysis_image, (x, y),
                          (x + w, y + h), (255, 0, 0), 2)
            shape_imgs = analysis_image
            if self.parameters.debug == "print":
                pcv.print_image(
                    shape_imgs,
                    filename=self.name_save + "_shape_analysis.JPG",
                )
            boundary_img1 = self.original_img.copy()
            cv2.line(boundary_img1, (0, 1680),
                     (self.original_img.shape[1], 1680), (0, 0, 255), 2)
            if self.parameters.debug == "print":
                pcv.print_image(
                    boundary_img1,
                    filename=self.name_save + "_boundary_analysis.JPG",
                )
            chans = cv2.split(self.original_img)
            colors = ("b", "g", "r")
            hist_canvas = np.zeros((300, 256, 3), dtype="uint8")
            for (chan, color) in zip(chans, colors):
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
                        hist_canvas, (x, 300), (x, 300 - int(y)),
                        {
                            "b": (255, 0, 0),
                            "g": (0, 255, 0),
                            "r": (0, 0, 255)
                        }[color],
                        1,
                    )
            if self.parameters.debug == "print":
                pcv.print_image(
                    hist_canvas,
                    filename=self.name_save + "_color_analysis.JPG",
                )
        self.mask_img = mask
        self.detected_obj = obj
        return analysis_image

    def generate_pseudolandmarks(self):
        """
        Generate pseudolandmarks for the image.
        """
        if self.mask_img is None:
            self.analyze_objects()
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=self.original_img, mask=self.mask_img,
            label=pcv.params.sample_label)
        if self.parameters.debug == "print":
            annotated = self.original_img.copy()
            for pt in top:
                pt_arr = np.asarray(pt).flatten()
                cv2.circle(
                    annotated, (int(
                        pt_arr[0]), int(
                        pt_arr[1])), 3, (255, 0, 0), -1)
            for pt in center_v:
                pt_arr = np.asarray(pt).flatten()
                cv2.circle(
                    annotated, (int(
                        pt_arr[0]), int(
                        pt_arr[1])), 3, (0, 69, 255), -1)
            for pt in bottom:
                pt_arr = np.asarray(pt).flatten()
                cv2.circle(
                    annotated, (int(
                        pt_arr[0]), int(
                        pt_arr[1])), 3, (255, 19, 240), -1)
            pcv.print_image(
                annotated,
                filename=self.name_save + "_pseudolandmarks.JPG")
        self.top = top
        self.bottom = bottom
        self.center_v = center_v
        return top, bottom, center_v

    def generate_color_histogram(self):
        """
        Generate an overlapping color histogram for the image with 9 channels:
        Blue, Green, Red, Lightness, Hue, Saturation, Value, Green-Magenta,
        and Blue-Yellow.
        """
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
            {"label": "Blue", "img": b, "bins": 256, "color": (255, 0, 0)},
            {"label": "Green", "img": g, "bins": 256, "color": (0, 255, 0)},
            {"label": "Red", "img": r, "bins": 256, "color": (0, 0, 255)},
            {"label": "Lightness", "img": lab_L, "bins": 256, "color": (128, 128, 128)},
            {"label": "Hue", "img": hsv_h, "bins": 180, "color": (0, 165, 255)},
            {"label": "Saturation", "img": hsv_s, "bins": 256, "color": (75, 0, 130)},
            {"label": "Value", "img": hsv_v, "bins": 256, "color": (255, 165, 0)},
            {"label": "Green-Magenta", "img": lab_a, "bins": 256, "color": (255, 105, 180)},
            {"label": "Blue-Yellow", "img": lab_b, "bins": 256, "color": (0, 255, 255)},
        ]

        for ch in channel_list:
            bins = ch["bins"]
            hist = cv2.calcHist([ch["img"]], [0], None, [bins], [0, bins])
            cv2.normalize(
                hist,
                hist,
                alpha=0,
                beta=canvas_height,
                norm_type=cv2.NORM_MINMAX)
            hist = hist.flatten()
            scale = canvas_width / bins
            for i in range(1, bins):
                x1 = int((i - 1) * scale)
                x2 = int(i * scale)
                y1 = canvas_height - int(hist[i - 1])
                y2 = canvas_height - int(hist[i])
                cv2.line(canvas, (x1, y1), (x2, y2), ch["color"], 1)
        if self.parameters.debug == "print":
            pcv.print_image(
                canvas,
                filename=self.name_save +
                "_color_histogram.JPG")
        return canvas
