import os
import numpy as np
from pydantic import BaseModel, DirectoryPath
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

__all__ = ['HyperspectralPCA', 'plot_hsi_channels' , 'plot_hyperspectral_histograms' ,
           'plot_masks_with_thresholds' , 'create_hsi_masks' , 'keep_largest_connected_component']

class HyperspectralPCA(BaseModel):
    n_components: int
    source_dir: DirectoryPath
    save_dir: str   # may not exist initially

    def perform_pca_on_image(self, image_path: str):
        """
        Perform PCA on a single hyperspectral image.

        Parameters:
        - image_path (str): Path to the .npy file containing the image.

        Returns:
        - pca_image (numpy.ndarray): Image with reduced number of components after PCA.
        - pca: The PCA object to extract explained variance.
        """
        # Step 1: Load the image from the .npy file
        image = np.load(image_path)

        # Step 2: Reshape the image from (H, W, C) to (H*W, C)
        H, W, C = image.shape
        image_reshaped = image.reshape(-1, C)

        # Step 3: Standardize the data (zero mean and unit variance)
        scaler = StandardScaler()
        image_standardized = scaler.fit_transform(image_reshaped)

        # Step 4: Apply PCA
        pca = PCA(n_components=self.n_components)
        image_pca = pca.fit_transform(image_standardized)

        # Step 5: Reshape back to the original image shape but with reduced components
        pca_image = image_pca.reshape(H, W, self.n_components)

        return pca_image, pca

    def perform_pca_on_directory(self):
        """
        Perform PCA on all .npy files in the source directory and save them to the save directory.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for file_name in os.listdir(self.source_dir):
            if file_name.endswith(".npy"):
                image_path = os.path.join(self.source_dir, file_name)
                pca_image, _ = self.perform_pca_on_image(image_path)

                # Save the PCA-reduced image with the same name to the save directory
                save_path = os.path.join(self.save_dir, file_name)
                np.save(save_path, pca_image)
                print(f"Saved PCA image to {save_path}")

    def generate_scree_plot(self, n_samples=30):
        """
        Sample 30 images from the source directory, perform PCA, and save a scree plot.

        Parameters:
        - n_samples (int): Number of images to sample from the source directory.
        """
        file_names = [f for f in os.listdir(self.source_dir) if f.endswith(".npy")]

        # Sample 30 random images from the directory
        sampled_files = random.sample(file_names, min(n_samples, len(file_names)))

        fig, axs = plt.subplots(5, 6, figsize=(20, 15))  # 5 rows x 6 columns for scree plots
        axs = axs.ravel()  # Flatten for easier iteration

        for idx, file_name in enumerate(sampled_files):
            image_path = os.path.join(self.source_dir, file_name)

            # Perform PCA on the sampled image
            _, pca = self.perform_pca_on_image(image_path)

            # Plot the scree plot (eigenvalue plot)
            explained_variance = pca.explained_variance_ratio_
            axs[idx].plot(np.cumsum(explained_variance), marker='o')
            axs[idx].set_title(file_name)
            axs[idx].set_xlabel("Number of Components")
            axs[idx].set_ylabel("Cumulative Explained Variance")

        # Adjust layout and save the plot
        plt.tight_layout()
        scree_plot_path = os.path.join(self.save_dir, "scree_plot.png")
        plt.savefig(scree_plot_path)
        print(f"Scree plot saved to {scree_plot_path}")



def plot_hsi_channels(directory: str, save_path: str):
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    if not npy_files:
        print("No .npy files found in the directory.")
        return

    # Select a random file
    # random_file = random.choice(npy_files)
    random_file = '1_85.npy'
    file_path = os.path.join(directory, random_file)

    # Load the .npy file
    image = np.load(file_path)

    if len(image.shape) != 3:
        print("The loaded file does not have the expected 3-dimensional shape.")
        return

    # Number of channels
    num_channels = image.shape[2]

    # Determine the number of rows needed for 15 columns
    num_cols = 15
    num_rows = (num_channels + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    # Plot each channel
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(image[:, :, i], cmap='gray')  # Use 'gray' colormap for dark background
        ax.set_title(f'Channel {i}')
        ax.axis('off')

    # Turn off any unused subplots
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')

    # Set the super title
    plt.suptitle(f'Hyperspectral Image Channels\nSelected File: {random_file}', fontsize=16)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the bottom to make space for the super title
    plt.savefig(save_path)
    plt.close()

def plot_hyperspectral_histograms(dir_path,  save_dir,n=3):
    # List all .npy files in the directory
    files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]

    # Randomly select n files from the directory
    # selected_files = random.sample(files, n)
    selected_files = ['5_39.npy' , '5_108.npy' , '21_13.npy' , '21_77.npy' , '89_108.npy']

    # Iterate over selected files
    for idx, file in tqdm(enumerate(selected_files), desc='Plotting Pixel Histograms of Hyperspectral Images'):
        # Load the hyperspectral image
        image = np.load(os.path.join(dir_path, file))

        # Number of channels
        num_channels = image.shape[2]  # Assuming the shape is (height, width, channels)

        # Create subplots for the image's channels
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, num_channels * 3))
        fig.suptitle(f'Histograms for {file}', fontsize=16)

        # Iterate over each channel
        for i in range(num_channels):
            ax = axes[i]
            channel_data = image[:, :, i].ravel()  # Flatten the channel data
            ax.hist(channel_data, bins=50, color='blue', alpha=0.7)
            ax.set_title(f'Channel {i+1}')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title

        # Save the plot
        output_file = os.path.join(save_dir, f'histograms_masked_{file[:-4]}.png')
        plt.savefig(output_file)
        plt.close(fig)  # Close the figure to free up memory


def plot_masks_with_thresholds(dir_path, save_dir, n=3, thresholds=np.arange(0.01, 0.4, 0.03)):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # List all .npy files in the directory
    files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]

    # Randomly select n files from the directory
    selected_files = random.sample(files, n)

    # Number of thresholds and images
    num_thresholds = len(thresholds)
    num_images = len(selected_files)

    # Create subplots for all images and thresholds
    num_cols = num_thresholds + 1  # +1 for the original image
    num_rows = num_images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    fig.suptitle('Original and Masks at Various Thresholds', fontsize=16)

    # Flatten axes array for easier indexing
    axes = axes.flatten() if num_images > 1 else [axes]

    # Iterate over selected files and thresholds
    for img_idx, file in tqdm(enumerate(selected_files) , desc='Plotting Masks with Thresholds'):
        # Load the hyperspectral image
        image = np.load(os.path.join(dir_path, file))

        # Assume the image has at least one channel and use the first channel for masking
        channel_data = image[:, :, 50]  # You might need to adjust this depending on your data

        # Plot the original image in the first column
        original_ax = axes[img_idx * num_cols]
        original_ax.imshow(channel_data, cmap='gray', vmin=np.min(channel_data), vmax=np.max(channel_data))
        original_ax.set_title('Original')
        original_ax.axis('off')

        # Create binary masks for each threshold and plot them
        for thresh_idx, threshold in enumerate(thresholds):
            mask = (channel_data > threshold).astype(np.float32)
            ax = axes[img_idx * num_cols + thresh_idx + 1]  # +1 for the threshold column
            ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Threshold {threshold:.2f}')
            ax.axis('off')

    # Hide any unused subplots (in case num_images * num_cols < total number of axes)
    for j in range(num_images * num_cols, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title

    # Save the plot
    output_file = os.path.join(save_dir, 'masks_with_thresholds.png')
    plt.savefig(output_file)
    plt.close(fig)  # Close the figure to free up memory


from skimage.morphology import binary_erosion, binary_dilation

def create_hsi_masks(input_dir, output_dir, threshold_value, channel_index=49, erosion_size=2, dilation_size=2):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each .npy file in the directory
    for filename in tqdm(os.listdir(input_dir), desc="Creating Masks for Hyperspectral Images"):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the .npy file
            image = np.load(file_path)

            # Extract the 50th channel (index 49, since Python is 0-based)
            channel = image[:, :, channel_index]

            # Threshold the channel to create a binary mask
            binary_mask = (channel >= threshold_value).astype(np.uint8)

            # Perform erosion on the binary mask
            eroded_mask = binary_erosion(binary_mask, footprint=np.ones((erosion_size, erosion_size)))
            eroded_mask = (eroded_mask * 255).astype(np.uint8)

            # Keep only the largest connected component after erosion
            largest_component_mask = keep_largest_connected_component(eroded_mask)

            # Perform dilation after finding the largest connected component
            dilated_mask = binary_dilation(largest_component_mask, footprint=np.ones((dilation_size, dilation_size)))

            # Fill any holes inside the largest component
            filled_mask = binary_fill_holes(dilated_mask).astype(np.uint8) * 255

            # Save the resulting mask to the output directory
            np.save(output_path, filled_mask)


def create_hsi_masks(input_dir, output_dir, threshold_value, channel_indices=[49], erosion_size=2, dilation_size=2):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each .npy file in the directory
    for filename in tqdm(os.listdir(input_dir), desc="Creating Masks for Hyperspectral Images"):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the .npy file
            image = np.load(file_path)

            # Initialize combined mask
            combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

            # Process each channel index
            for channel_index in channel_indices:
                # Extract the specified channel
                channel = image[:, :, channel_index]

                # Threshold the channel to create a binary mask
                binary_mask = (channel >= threshold_value).astype(np.uint8)

                # Bitwise AND with the combined mask
                combined_mask = np.bitwise_and(combined_mask, binary_mask) if np.any(combined_mask) else binary_mask

            # Erode the binary mask
            eroded_mask = binary_erosion(combined_mask, footprint=np.ones((erosion_size, erosion_size)))
            eroded_mask = (eroded_mask * 255).astype(np.uint8)
            # Keep only the largest connected component after erosion
            largest_component_mask = keep_largest_connected_component(combined_mask)

            # Perform dilation after finding the largest connected component
            dilated_mask = binary_dilation(largest_component_mask, footprint=np.ones((dilation_size, dilation_size)))

            # Fill any holes inside the largest component
            filled_mask = binary_fill_holes(dilated_mask).astype(np.uint8) * 255

            # Save the resulting mask to the output directory
            np.save(output_path, filled_mask)




def keep_largest_connected_component(binary_mask):
    # Find all connected components (blobs) in the binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Get the area of each component and check for boundary touching components
    valid_components = []
    for label in range(1, num_labels):  # Skipping the background (label 0)
        x, y, w, h, area = stats[label]

        # Check if the component touches the boundary
        if x > 0 and y > 0 and (x + w) < binary_mask.shape[1] and (y + h) < binary_mask.shape[0]:
            valid_components.append((label, area))

    # Find the largest component that doesn't touch the boundary
    if valid_components:
        largest_component_label = max(valid_components, key=lambda x: x[1])[0]

        # Create a mask for the largest valid component
        largest_component_mask = np.zeros_like(binary_mask)
        largest_component_mask[labels == largest_component_label] = 255

        return largest_component_mask
    else:
        # If no valid components are found, return an empty mask
        return np.zeros_like(binary_mask)


def apply_mask_to_hyperspectral_images(mask_dir, hyper_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the mask directory
    mask_files = os.listdir(mask_dir)

    for mask_file in tqdm(mask_files , desc= 'Applying Masks to Hyperspectral Images'):
        # Construct the full file paths
        mask_path = os.path.join(mask_dir, mask_file)
        hyper_path = os.path.join(hyper_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)

        # Load the mask and hyperspectral image
        mask = np.load(mask_path).astype(bool)
        hyper_image = np.load(hyper_path)

        # Apply the mask to each channel of the hyperspectral image
        masked_image = hyper_image * mask[..., np.newaxis]

        # Save the masked hyperspectral image
        np.save(output_path, masked_image)



if __name__ == '__main__':
    # Example usage
    # pca = HyperspectralPCA(n_components=30, source_dir='Data/hsi', save_dir='Data/pca_images')
    # pca.perform_pca_on_directory()
    # pca.generate_scree_plot()

    plot_hsi_channels('Data/hsi_masked_trimmed', 'pics/hsi_channels_masked.png')
    # plot_hsi_channels('Data/pca_images', 'Data/pca_channels.png')

    # Example usage
    # plot_hyperspectral_histograms('Data/hsi_masked',save_dir='pics' ,  n=5)

    # plot_masks_with_thresholds(dir_path='Data/hsi', save_dir='pics', n=5, thresholds=np.arange(0.01, 0.14, 0.01))
    # create_hsi_masks(input_dir='Data/hsi', output_dir='Data/hsi_seed_masks', threshold_value=0.12 ,
    #                  channel_indices=[30, 35 , 40 , 50 , 60, 70 ,80] ,erosion_size=5, dilation_size=4)


    # create_random_subplots(input_dir='Data/hsi_seed_masks', save_dir='pics', num_images=16)
    # apply_mask_to_hyperspectral_images(mask_dir='Data/hsi_seed_masks', hyper_dir='Data/hsi_trimmed', output_dir='Data/hsi_masked_trimmed')
    pass