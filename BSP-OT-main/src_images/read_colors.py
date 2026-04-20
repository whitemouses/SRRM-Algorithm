import cv2
import numpy as np
import argparse

def write_pixel_colors(image_path, output_file):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    height, width, _ = image.shape
    with open(output_file, 'w') as f:
        for y in range(height):
            for x in range(width):
                pixel = image[y, x]
                f.write(f"{pixel[0]} {pixel[1]} {pixel[2]}\n")

    print(f"Pixel colors written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pixel colors from an image and write to a file.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("output_file", help="Path to the output text file.")

    args = parser.parse_args()

    write_pixel_colors(args.image_path, args.output_file)

