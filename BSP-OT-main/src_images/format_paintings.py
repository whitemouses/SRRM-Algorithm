import os
import cv2
import numpy as np
import argparse

def get_average_color(image):
    return np.mean(image, axis=(0, 1))  # Average across height and width

def process_images(input_folder, output_folder, output_file, size):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(output_file, 'w') as f:
        image_id = 0
        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    resized_image = cv2.resize(image, size)
                    avg_color = get_average_color(resized_image)
                    
                    output_image_path = os.path.join(output_folder, f"{image_id}.jpg")
                    cv2.imwrite(output_image_path, resized_image)
                    
                    f.write(f"{image_id}: {avg_color}\n")
                    image_id += 1
                else:
                    print(f"Skipping {filename}: Unable to read image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images and compute average colors.")
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("output_folder", help="Path to the folder where resized images will be stored.")
    parser.add_argument("output_file", help="Path to the output text file to store average colors.")
    parser.add_argument("--width", type=int, default=100, help="Width of resized images.")
    parser.add_argument("--height", type=int, default=100, help="Height of resized images.")
    
    args = parser.parse_args()
    size = (args.width, args.height)
    
    process_images(args.input_folder, args.output_folder, args.output_file, size)
    print(f"Processing complete. Resized images saved to {args.output_folder}, average colors saved to {args.output_file}")

