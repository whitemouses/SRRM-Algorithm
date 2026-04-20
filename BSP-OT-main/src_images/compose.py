import os
import cv2
import numpy as np
import argparse

def create_image_from_list(image_folder, number_file, output_image):
    with open(number_file, 'r') as f:
        indices = [int(line.strip()) for line in f]

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))], key=lambda x: int(os.path.splitext(x)[0]))


    if not image_files:
        print("Error: No images found in the folder.")
        return

    sample_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    if sample_image is None:
        print("Error: Unable to read sample image.")
        return

    block_size = sample_image.shape[0]
    grid_size = int(np.ceil(np.sqrt(len(indices))))
    new_image_size = grid_size * block_size
    new_image = np.zeros((new_image_size, new_image_size, 3), dtype=np.uint8)
    print("each block size {} grid size {} new image size {}".format(block_size, grid_size, new_image_size))

    for idx, img_index in enumerate(indices):
        if img_index < 0 or img_index >= len(image_files):
            print(f"Warning: Index {img_index} out of range, skipping.")
            continue

        img = cv2.imread(os.path.join(image_folder, image_files[idx]))
        if img is None:
            print(f"Warning: Unable to read image {image_files[img_index]}, skipping.")
            continue

        row, col = divmod(img_index, grid_size)
        new_image[row * block_size: (row + 1) * block_size, col * block_size: (col + 1) * block_size] = img

    cv2.imwrite(output_image, new_image)
    print(f"New image saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an image by arranging blocks based on a list of indices.")
    parser.add_argument("image_folder", help="Path to the folder containing images.")
    parser.add_argument("number_file", help="Path to the file containing list of numbers.")
    parser.add_argument("output_image", help="Path to save the output image.")

    args = parser.parse_args()

    create_image_from_list(args.image_folder, args.number_file, args.output_image)

