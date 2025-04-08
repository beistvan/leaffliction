
import os
import sys
import cv2
import argparse
import numpy as np


def load_image(image_path):
    """Load an image (and convert from BGR to RGB) using OpenCV."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image(image, save_path):
    """Save an RGB image to file (convert to BGR first)."""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)


def transformation_original(image):
    """Return the original image."""
    return image.copy()


def transformation_gaussian(image):
    """Apply a Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (7, 7), 0)


def transformation_mask(image):
    """
    Create a binary mask based on HSV thresholds.
    (For example, to roughly segment a green leaf.)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask_rgb


def transformation_roi(image):
    """
    Find contours from the mask and draw bounding boxes (ROIs)
    on the image.
    """
    mask = cv2.cvtColor(transformation_mask(image), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_roi = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image_roi


def transformation_analyze(image):
    """
    Find the largest contour, compute its area and perimeter,
    and annotate the image.
    """
    mask = cv2.cvtColor(transformation_mask(image), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def transformation_pseudolandmarks(image):
    """
    Detect corners (pseudolandmarks) using goodFeaturesToTrack and mark them.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    image_landmarks = image.copy()
    if corners is not None:
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image_landmarks, (x, y), 3,
                       (255, 0, 255), -1)
    return image_landmarks


def plot_color_histogram(image, ax):
    """Plot the color histogram (for each channel) on the provided axis."""
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title("Color Histogram")


def process_image(image_path):
    """Load an image and compute all transformations."""
    image = load_image(image_path)
    transformations = {
        "Original": transformation_original(image),
        "Gaussian_blur": transformation_gaussian(image),
        "Mask": transformation_mask(image),
        "ROI_objects": transformation_roi(image),
        "Analyze_object": transformation_analyze(image),
        "Pseudolandmarks": transformation_pseudolandmarks(image)
    }
    return image, transformations


def display_transformations(image, transformations):
    """
    Display all seven figures (six transformations plus color histogram)
    in one matplotlib window.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib or NumPy not available. Skipping GUI display.")
        return

    fig = plt.figure(figsize=(12, 10))
    titles = [
        "Figure IV.1: Original",
        "Figure IV.2: Gaussian blur",
        "Figure IV.3: Mask",
        "Figure IV.4: ROI objects",
        "Figure IV.5: Analyze object",
        "Figure IV.6: Pseudolandmarks",
        "Figure IV.7: Color histogram"
    ]

    trans_order = [
        transformations["Original"],
        transformations["Gaussian_blur"],
        transformations["Mask"],
        transformations["ROI_objects"],
        transformations["Analyze_object"],
        transformations["Pseudolandmarks"]
    ]

    for i, img in enumerate(trans_order):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')

    ax = fig.add_subplot(3, 3, 7)
    plot_color_histogram(image, ax)
    ax.axis('on')

    plt.tight_layout()
    plt.show()


def save_transformations(image_path, dst_dir):
    """
    Process one image and save the resulting transformations in the
    destination directory. The color histogram is saved as a PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib or NumPy not available. Skipping histogram save.")

        image, transformations = process_image(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(dst_dir, exist_ok=True)
        for name, img in transformations.items():
            save_path = os.path.join(dst_dir, f"{base_name}_{name}.jpg")
            save_image(img, save_path)
            print(f"Saved: {save_path}")
        return

    image, transformations = process_image(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs(dst_dir, exist_ok=True)

    for name, img in transformations.items():
        save_path = os.path.join(dst_dir, f"{base_name}_{name}.jpg")
        save_image(img, save_path)
        print(f"Saved: {save_path}")

    hist_path = os.path.join(dst_dir, f"{base_name}_Color_histogram.png")
    fig, ax = plt.subplots()
    plot_color_histogram(image, ax)
    plt.savefig(hist_path)
    plt.close(fig)
    print(f"Saved: {hist_path}")


def process_directory(src_dir, dst_dir):
    """
    Process all image files in the source directory and save the
    transformations to the destination directory.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for entry in os.scandir(src_dir):
        if entry.is_file() and os.path.splitext(
                entry.name)[1].lower() in valid_extensions:
            save_transformations(entry.path, dst_dir)


def main():
    parser = argparse.ArgumentParser(
        description="""Image Transformation - Apply several
image-processing operations on a single image or all images in a directory."""
    )

    parser.add_argument("path", nargs="?", help="Path to an image file")

    parser.add_argument("-src", help="Source directory containing images")
    parser.add_argument(
        "-dst",
        help="Destination directory to save transformed images")
    parser.add_argument(
        "-mask",
        action="store_true",
        help="Flag for additional mask processing (if needed)")
    parser.add_argument(
        "-headless",
        action="store_true",
        help="Disable the GUI display (useful for headless environments)")

    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
        print("No display found. Running in headless mode.")
        args.headless = True
    if args.headless:
        import matplotlib
        matplotlib.use("Agg")

    if args.src:
        if not args.dst:
            print("Error: In directory mode, -dst must be provided ", end="")
            print("to save output images.")
            sys.exit(1)
        if not os.path.isdir(args.src):
            print(f"Error: {args.src} is not a valid directory.")
            sys.exit(1)
        process_directory(args.src, args.dst)
    elif args.path:
        if not os.path.isfile(args.path):
            print(f"Error: {args.path} is not a valid file.")
            sys.exit(1)
        image, transformations = process_image(args.path)
        if args.headless:
            base_dir = args.dst if args.dst else "output"
            os.makedirs(base_dir, exist_ok=True)
            save_transformations(args.path, base_dir)
        else:
            display_transformations(image, transformations)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
