# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#   Distribution.py                                    :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2025/03/31 11:12:34 by iberegsz          #+#    #+#              #
#   Updated: 2025/04/02 21:10:05 by iberegsz         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def is_image_file(filename):
    """Check if a file is an image based on its extension.
    Args:
        filename (str): The name of the file to check.

    Raises:
        ValueError: If the filename is empty or None.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    try:
        if not filename:
            raise ValueError("Filename cannot be empty or None.")
        valid_extensions = {
            '.jpg',
            '.jpeg',
            '.png',
            '.gif',
            '.bmp',
            '.tif',
            '.tiff'}
        return os.path.splitext(filename.lower())[1] in valid_extensions
    except Exception as e:
        print(f"Error in is_image_file: {e}")
        raise


def main():
    """Main function to analyze the distribution of images in subdirectories.

    Args:
        None

    Raises:
        SystemExit: If the script is not run with the correct arguments.

    Returns:
        None
    """
    try:
        if len(sys.argv) < 2:
            print(f"Usage: {sys.argv[0]} <path_to_dataset>")
            sys.exit(1)

        top_level_dir = sys.argv[1]
        if not os.path.isdir(top_level_dir):
            raise NotADirectoryError(
                f"{top_level_dir} is not a valid directory.")

        dataset_name = os.path.basename(os.path.abspath(top_level_dir))
        distribution = {}

        for entry in os.scandir(top_level_dir):
            if entry.is_dir():
                try:
                    subdir_name = os.path.basename(entry.path)
                    dataset_prefix = dataset_name.lower() + "_"
                    if subdir_name.lower().startswith(dataset_prefix):
                        subdir_name = subdir_name[len(dataset_prefix):]

                    image_count = sum(1 for _, _, files in os.walk(entry.path)
                                      for f in files if is_image_file(f))
                    distribution[subdir_name] = image_count
                except Exception as e:
                    print(f"Error processing directory {entry.path}: {e}")

        if not distribution:
            print(f"No valid image subdirectories found in {top_level_dir}.")
            sys.exit(0)

        labels = list(distribution.keys())
        counts = list(distribution.values())

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors)
        plt.title(f"{dataset_name} Distribution (Pie)")

        plt.subplot(1, 2, 2)
        plt.bar(labels, counts, color=colors)
        plt.title(f"{dataset_name} Distribution (Bar)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.show()
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
