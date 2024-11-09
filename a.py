import os
import numpy as np
from PIL import Image

# Define paths
input_dir = 'Data/hsi_seed_masks'  # Replace with the path to your .npy files
output_dir = 'Data/shyam_sundar'  # Replace with the path to save .jpg files



# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert binary mask to RGB
def binary_to_rgb(binary_mask):
    # Ensure binary_mask contains values 0 and 1, then scale to 0-255
    rgb_mask = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    return rgb_mask.astype(np.uint8)

# Iterate through .npy files in the input directory
for i in range(4):
  for j in range(144) :
        npy_file = f"{i}_{j}.npy"
        mask = np.load(os.path.join(input_dir, npy_file))
        # Ensure mask is binary and has the correct dimensions
        if mask.ndim == 2:  # Ensure it's 2D
            # Print min and max values to check if it's truly binary
            print(f'Processing {npy_file} - Min value: {mask.min()}, Max value: {mask.max()}')

            # Convert mask to RGB
            rgb_mask = binary_to_rgb(mask)
            print(min(mask.flatten()), max(mask.flatten()))

            # Save as .jpg file
            output_file = os.path.join(output_dir, npy_file.replace('.npy', '.jpg'))
            Image.fromarray(rgb_mask).save(output_file)
            print(f'Saved {output_file}')
        else:
            print(f'Skipping {npy_file} - Incorrect dimensions')