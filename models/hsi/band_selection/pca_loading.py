import os
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import pickle
class PCAChannelSelector:

    def __init__(self, source_dir, dest_dir, components_list, n_sample_images=2000, n_components=50):
        """
        Initialize the PCAChannelSelector.

        Parameters:
        source_dir (str): The directory where the source images are stored.
        dest_dir (str): The directory where processed images will be saved.
        components_list (list): A list specifying the number of top PCA channels to select.
        n_sample_images (int): The number of random images to sample for PCA.
        n_components (int): The number of PCA components to compute.
        """
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.components_list = components_list
        self.n_sample_images = n_sample_images
        self.n_components = n_components
        self.global_loadings = None  # Will store global PCA loadings

    def sample_images(self):
        """
        Sample n_sample_images from the source directory.

        Returns:
        sampled_files (list): A list of sampled image file paths.
        """
        image_files = [f for f in os.listdir(self.source_dir) if f.endswith('.npy')]

        # Ensure we do not sample more images than available
        n_images = min(self.n_sample_images, len(image_files))
        sampled_files = np.random.choice(image_files, n_images, replace=False)

        return sampled_files

    def aggregate_sampled_data(self, sampled_files):
        """
        Aggregate pixel data across the sampled images.

        Parameters:
        sampled_files (list): List of sampled image file names.

        Returns:
        aggregated_data (np.ndarray): The aggregated pixel data from the sampled images.
        """
        aggregated_data = []

        for image_file in tqdm(sampled_files, desc="Aggregating Data"):
            image_data = np.load(os.path.join(self.source_dir, image_file))  # Load image data
            H, W, C = image_data.shape
            reshaped_data = image_data.reshape(-1, C)  # Reshape to (H*W, C)
            aggregated_data.append(reshaped_data)

        return np.vstack(aggregated_data)  # Stack all image data into one large array

    def apply_pca(self, data):
        """
        Apply PCA on aggregated data and calculate global loadings.

        Parameters:
        data (np.ndarray): Aggregated pixel data of shape (N, C), where N is the total number of pixels and C is the number of channels.
        """
        pca = PCA(n_components=self.n_components)
        pca.fit(data)  # Fit PCA on the aggregated sampled data
        self.global_loadings = pca.components_.T  # Store the global PCA loadings (shape: C x n_components)


    def rank_channels(self):
        """
        Rank the channels based on the global PCA loadings and save the ranking with corresponding weights.

        Returns:
        ranking_dict (dict): A dictionary with 'channel_ranking' and 'channel_weightage' lists.
        """
        # Sum of absolute PCA loadings across components to determine channel importance
        channel_importance = np.sum(np.abs(self.global_loadings), axis=1)

        # Rank channels based on their importance
        ranked_indices = np.argsort(channel_importance)[::-1]  # Sort in descending order

        # Create a dictionary with separate lists for ranking and corresponding weights
        ranking_dict = {
            'channel_ranking': ranked_indices.tolist(),
            'channel_weightage': channel_importance[ranked_indices].tolist()
        }

        # Save the dictionary to a pickle file
        with open("models/hsi/band_selection/pcaLoading_ranked_channels.pkl", "wb") as f:
            pickle.dump(ranking_dict, f)

        return ranking_dict


    def process_image(self, image_file, selected_channels):
        """
        Process a single image by selecting the top channels and saving the result.

        Parameters:
        image_file (str): Path to the image file.
        selected_channels (list): Indices of the selected channels to retain.

        Returns:
        np.ndarray: Processed image with selected channels.
        """
        image_data = np.load(image_file)  # Assuming images are saved as .npy files (H, W, C)
        selected_data = image_data[:, :, selected_channels]  # Select the top channels from the original image

        return selected_data

    def save_image(self, data, save_dir, image_file):
        """
        Save the processed image with selected channels in the given directory.

        Parameters:
        data (np.ndarray): The processed image data.
        save_dir (str): Directory where the processed image will be saved.
        image_file (str): The original image file name.
        """
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = os.path.join(save_dir, os.path.basename(image_file))
        np.save(save_path, data)  # Save the processed image data

    def apply_band_selection(self):
        """
        Apply band selection to all images in the source directory based on PCA rankings.
        """
        # Step 1: Sample n_sample_images from the source directory
        sampled_files = self.sample_images()

        # Step 2: Aggregate pixel data from the sampled images
        aggregated_data = self.aggregate_sampled_data(sampled_files)

        # Step 3: Apply PCA on the aggregated sampled data to compute loadings
        self.apply_pca(aggregated_data)

        # Step 4: Rank channels based on PCA loadings
        ranked_channels = self.rank_channels()

        # Step 5: Process all images in the source directory based on the ranked channels
        image_files = [f for f in os.listdir(self.source_dir) if f.endswith('.npy')]

        for n_channels in self.components_list:

            # Select the top n_channels from the ranked list
            selected_channels = ranked_channels[:n_channels]

            for image_file in tqdm(image_files, desc=f"Processing for {n_channels} channels"):
                image_path = os.path.join(self.source_dir, image_file)
                selected_data = self.process_image(image_path, selected_channels)

                # Save the processed image in a subdirectory named "Channel_{n_channels}"
                save_dir = os.path.join(self.dest_dir, f"Channel_{n_channels}")
                self.save_image(selected_data, save_dir, image_file)


if __name__=="__main__":
    pca_loader = PCAChannelSelector(source_dir='Data/hsi_masked', dest_dir="Data/hsi_masked_pcaLoading",
                                    components_list=[25,50,75,100,125,150,168])

    pca_loader.apply_band_selection()