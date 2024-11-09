import numpy as np
import os
from glob import glob
from tqdm import tqdm
import pickle

class BandSelectorSPA:
    def __init__(self, hsi_image_dir: str, dest_dir: str, top_channels: list, batch_size: int = 100):
        """
        Initialize BandSelectorSPA.

        Parameters:
        hsi_image_dir (str): Directory where HSI images are stored.
        dest_dir (str): Directory to save the results.
        top_channels (list): List of integers specifying the number of top channels to extract (e.g., [20, 50, 100]).
        batch_size (int): Number of images to process in a batch for memory efficiency.
        """
        self.hsi_image_dir = hsi_image_dir
        self.dest_dir = dest_dir
        self.top_channels = top_channels
        self.batch_size = batch_size

        os.makedirs(self.dest_dir, exist_ok=True)

    def load_hsi_image(self, image_path: str):
        """
        Load a hyperspectral image from a given path and transpose it to (C, H, W).

        Parameters:
        image_path (str): Path to the HSI image.

        Returns:
        np.ndarray: Loaded HSI image (C, H, W).
        """
        image = np.load(image_path)
        return np.transpose(image, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)

    def save_hsi_image(self, image: np.ndarray, save_path: str):
        """
        Save the hyperspectral image to a given path.

        Parameters:
        image (np.ndarray): HSI image to be saved (C, H, W).
        save_path (str): Path to save the HSI image.
        """
        np.save(save_path, np.transpose(image, (1, 2, 0)))  # Change from (C, H, W) to (H, W, C)

    def summarize_spectra(self, batch_data):
        """
        Summarize the spectral data for a batch of images.

        Parameters:
        batch_data (list): List of HSI images.

        Returns:
        np.ndarray: Mean spectrum for each image in the batch (batch_size, C).
        """
        summarized_data = []
        for img in batch_data:
            masked_pixels = img[:, img.any(axis=0)]  # Extract non-background pixels
            if masked_pixels.size == 0:  # Handle cases where no valid pixels are found
                continue
            mean_spectrum = np.mean(masked_pixels, axis=1)  # Mean across pixels
            summarized_data.append(mean_spectrum)

        if len(summarized_data) == 0:
            return np.array([])  # Return an empty array if no valid data

        return np.stack(summarized_data)  # Shape: (batch_size, C)

    def rank_full_channels(self):
        """
        Rank all channels across all images in the source directory using mean projection.

        Saves:
        A dictionary containing 'channel_ranking' and 'channel_weightage' lists.

        Returns:
        ranking_dict (dict): Dictionary containing ranked channels and their corresponding weights.
        """
        image_paths = glob(os.path.join(self.hsi_image_dir, "*.npy"))
        combined_data = []

        print(f"Ranking full channels for {len(image_paths)} images...")

        # Process images in batches
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing Batches for Ranking"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_data = [self.load_hsi_image(path) for path in batch_paths]
            summarized_data = self.summarize_spectra(batch_data)

            if summarized_data.size > 0:  # Ensure summarized data is valid
                combined_data.append(summarized_data)

        if len(combined_data) == 0:
            raise ValueError("No valid data found across all images.")

        # Concatenate all batch data (N_images, C) where C is the number of channels
        combined_data = np.concatenate(combined_data, axis=0)

        # Calculate channel ranking using the norm of projections
        projections = np.dot(combined_data.T, combined_data)  # (C, C) correlation matrix
        norms = np.linalg.norm(projections, axis=0)

        # Handle any NaN values in norms
        norms = np.nan_to_num(norms, nan=0.0)

        # Normalize norms to a smaller range, e.g., -2 to 2
        min_norm = np.min(norms)
        max_norm = np.max(norms)
        scaled_norms = 4 * (norms - min_norm) / (max_norm - min_norm) - 2  # Scale to range [-2, 2]

        # Rank channels based on the scaled norms
        ranked_channels = np.argsort(-scaled_norms)  # Sort channels by decreasing scaled norm

        # Create a dictionary with channel rankings and corresponding scaled norms (weights)
        ranking_dict = {
            'channel_ranking': ranked_channels.tolist(),
            'channel_weightage': scaled_norms[ranked_channels].tolist()
        }

        # Save the dictionary to a pickle file
        with open("models/hsi/band_selection/spa_ranked_channels.pkl", "wb") as f:
            pickle.dump(ranking_dict, f)

        return ranking_dict


    def run(self):
        """
        Run the BandSelectorSPA for ranking and channel selection.

        Returns:
        dict: A dictionary with keys as the number of top channels (e.g., 20, 50, 100) and values as the selected bands across all images.
        """
        # Step 1: Rank all channels
        ranked_channels = self.rank_full_channels()
        print("Full channel ranking --->\n", ranked_channels)

         # Save full channel ranking to the destination directory
        ranking_save_path = os.path.join(self.dest_dir, 'full_channel_ranking.npy')
        np.save(ranking_save_path, ranked_channels)
        print(f"Full channel ranking saved at {ranking_save_path}")

        # Step 2: Process all images and save selected channels
        self.process_all_images(ranked_channels)

        return ranked_channels


if __name__ == "__main__":
    # Usage
    hsi_selector = BandSelectorSPA(
        hsi_image_dir='Data/hsi_masked',
        dest_dir='Data/hsi_masked_spa',
        top_channels=[25, 50, 75, 100, 125, 150, 168],
        batch_size=100
    )
    ranked_channels = hsi_selector.run()
