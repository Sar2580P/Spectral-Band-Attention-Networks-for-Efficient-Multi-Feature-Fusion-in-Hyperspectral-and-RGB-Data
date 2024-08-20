import os
import numpy as np
from pydantic import BaseModel, DirectoryPath
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random


__all__ = ['HyperspectralPCA', 'plot_hsi_channels']

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
    random_file = random.choice(npy_files)
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