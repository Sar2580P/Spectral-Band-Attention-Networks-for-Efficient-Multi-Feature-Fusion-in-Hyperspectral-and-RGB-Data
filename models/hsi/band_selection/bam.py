from models.modules import Composite
from models.model_architectures import DenseNet, plot_model
from processing.utils import read_yaml
from torch import nn
import torch
import os
import numpy as np
import random
from tqdm import tqdm

class BandAttentionBlock(nn.Module):
    def __init__(self, in_channels, temperature=0.7, r=2, sparsity_threshold=0.07, apply_temperature_scaling=True):
        super(BandAttentionBlock, self).__init__()

        # Convolutional Layers
        self.conv_block = self._build_conv_block(in_channels)

        # Attention Mechanism
        self.attention_conv1d = self._build_attention_module(in_channels, r)

        # Learnable temperature scaling
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=apply_temperature_scaling)

        # Learnable sparsity threshold
        self.learnable_threshold = nn.Parameter(torch.tensor(sparsity_threshold))
        self.sigmoid = nn.Sigmoid()
        self.scaled_attn_vector = None
        config = read_yaml('models/hsi/config.yaml')
        self.apply_temperature_scaling = apply_temperature_scaling
        # plot_model(config = config , model = self.attention_conv1d)

    def _build_conv_block(self, in_channels):
        """
        Creates the sequential block with convolutional and pooling layers.
        """

        return nn.Sequential(
            Composite(in_channels=in_channels, out_channels=128, apply_BN=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Composite(in_channels=128, out_channels=96, apply_BN=False),
            Composite(in_channels=96, out_channels=64, apply_BN=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Composite(in_channels=64, out_channels=48, apply_BN=False),
            Composite(in_channels=48, out_channels=64 , apply_BN=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _build_attention_module(self, in_channels, reduction_ratio):
        """
        Creates the attention mechanism using Conv1d layers and BatchNorm.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=in_channels // reduction_ratio, kernel_size=32),
            nn.PReLU(),
            nn.Conv1d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=32),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # Attention path
        attn_vector = self.conv_block(x)  # Shape: (B, 32, 1, 1)
        attn_vector = attn_vector.squeeze(3).permute(0, 2, 1)  # Shape: (B, 1, 32)
        # Apply Conv1D Attention Mechanism
        attn_vector = self.attention_conv1d(attn_vector)  # (B, reduction_ratio, 1)
        if self.apply_temperature_scaling:
            self.scaled_attn_vector = attn_vector / self.temperature
        else :
            self.scaled_attn_vector = attn_vector

        # Sparsity enforcement
        sparse_channel_weights = torch.where(
            self.scaled_attn_vector  >= self.learnable_threshold,
            torch.sigmoid(self.scaled_attn_vector ),
            torch.zeros_like(attn_vector)
        )
        # Apply attention weights to the input
        output = x * sparse_channel_weights.unsqueeze(3)  # Shape: (B, C, H, W)
        return output

class BandMultiheadAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, r=2, temperature=0.1):
        super(BandMultiheadAttentionBlock, self).__init__()

        # Depthwise Separable Convolutional Block
        self.conv_block = self._build_conv_block(in_channels)

        # Multi-Head Attention
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([self._build_attention_module(in_channels, r) for _ in range(num_heads)])

        # Learnable temperature scaling
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Learnable sparsity threshold
        self.learnable_threshold = nn.Parameter(torch.tensor(0.5))

        # Activation
        self.sigmoid = nn.Sigmoid()
        self.scaled_attn_vector = None

    def _build_conv_block(self, in_channels):
        """
        Creates the sequential block with depthwise separable convolutional layers.
        """
        return nn.Sequential(
            self._depthwise_separable_conv(in_channels, 128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self._depthwise_separable_conv(128, 96),
            self._depthwise_separable_conv(96, 64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self._depthwise_separable_conv(64, 48),
            self._depthwise_separable_conv(48, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _depthwise_separable_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Depthwise separable convolution to reduce the number of parameters.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),  # Depthwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _build_attention_module(self, in_channels, reduction_ratio):
        """
        Creates the attention mechanism using Conv1d layers.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=in_channels // reduction_ratio, kernel_size=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=32),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        b, c, h, w = x.shape  # Input shape: (B, C, H, W)
        # Attention path through the convolutional block
        attn_vector = self.conv_block(x)  # Shape: (B, 64, 1, 1)
        attn_vector = attn_vector.squeeze(3).permute(0, 2, 1)  # Shape: (B, 1, 64)

        # Multi-Head Attention Mechanism
        attn_vectors = []
        for head in self.attention_heads:
            head_attn = head(attn_vector)  # Each head processes the input
            attn_vectors.append(head_attn)

        # Concatenate attention vectors from all heads
        attn_vector = torch.cat(attn_vectors, dim=1)  # Shape: (B, 4 * 147) if each head outputs (B, 147)
        # Apply temperature scaling to the attention weights
        self.scaled_attn_vector = attn_vector / self.temperature

        # Apply the learnable sparsity threshold
        sparse_channel_weights = torch.where(
            self.scaled_attn_vector >= self.learnable_threshold,
            self.sigmoid(self.scaled_attn_vector),  # Activate weights above threshold
            torch.zeros_like(self.scaled_attn_vector)  # Zero-out weights below threshold
        )

        # Reshape sparse_channel_weights for the number of heads
        sparse_channel_weights = sparse_channel_weights.view(b, self.num_heads, c)  # Shape: (B, num_heads, 147)
        # Aggregate across heads
        sparse_channel_weights = sparse_channel_weights.mean(dim=1)  # Shape: (B, 147)
        # Ensure the correct shape for multiplication
        sparse_channel_weights = sparse_channel_weights.unsqueeze(2).unsqueeze(3)  # Shape: (B, 147, 1, 1)
        output = x * sparse_channel_weights  # Ensure x has the shape (B, 147, H, W)

        return output



class BandAttentionIntegrated(nn.Module):
    def __init__(self, config):
        super(BandAttentionIntegrated, self).__init__()
        self.config = config
        self.model = self.get_model()
        self.model_name = config['model_name']
        self.layer_lr = [{'params' : self.bam.parameters(), 'lr' : self.config['lr']*0.5},
                         {'params' : self.head.parameters() , 'lr' : self.config['lr']}]

        # plot_model(self.config , self.model)
    def get_model(self):
        self.bam = BandAttentionBlock(in_channels=self.config['C'] ,temperature= self.config['sparse_bam_config']['temperature'],
                                      r = self.config['sparse_bam_config']['r'], sparsity_threshold=self.config['sparse_bam_config']['sparsity_threshold'],
                                      apply_temperature_scaling = self.config['sparse_bam_config']['apply_temperature_scaling'])
        # self.bam = BandMultiheadAttentionBlock(in_channels=self.config['C'] , r = self.config['sparse_bam_config']['r'],
        #                               num_heads = self.config['sparse_bam_config']['num_heads'],
        #                               temperature = self.config['sparse_bam_config']['temperature'])

        dup_config = self.config.copy()
        dup_config['model_name'] = self.config['sparse_bam_config']['head_model_name']   #the self.config['model_name'] is the name of the BAM integrated model
        if self.config['sparse_bam_config']['head_model_name'] == 'densenet':
            self.head = DenseNet(config=dup_config)


        self.load_head_ckpt()

        return nn.Sequential(self.bam, self.head)

    def load_head_ckpt(self):
        ckpt_file_path = self.config['sparse_bam_config']['head_model_ckpt']
        try:
            checkpoint = torch.load(ckpt_file_path, weights_only=True)['state_dict']

            weights = {k.replace("model_obj.model.", ""): v for k, v in checkpoint.items() if k.startswith("model_obj.model.")}
            self.head.model.load_state_dict(weights)
            print(f"Successfull loaded weights from {ckpt_file_path}")
        except Exception as e:
            print(f"Failed to load weights, initializing with random weights")

    def forward(self, x):
        return self.model(x)


def hook_fn(module, input, output):
    """
    Hook function to capture the scaled attention vector.
    """
    # Store the scaled attention vector in module's attribute
    module.attn_vector_output = module.scaled_attn_vector.detach().cpu()

def register_attention_hook(model):
    """
    Registers a forward hook on the BandAttentionBlock to capture the scaled attention vector.
    """
    # Register the hook on the BandAttentionBlock's forward method
    model.bam.register_forward_hook(hook_fn)

def get_scaled_attention_vector(model, input_tensor):
    """
    Passes input through the model and retrieves the scaled attention vector.

    Args:
        model: The BandAttentionIntegrated model (with loaded checkpoint).
        input_tensor: The input tensor (e.g., an image) to pass through the model.

    Returns:
        Scaled attention vector from the BandAttentionBlock.
    """
    # Make sure the attention hook is registered
    register_attention_hook(model)

    # Pass the input through the model
    with torch.no_grad():
        _ = model(input_tensor)

    # Get the scaled attention vector from the bam module
    return model.bam.attn_vector_output

def process_and_save_scaled_attention_vectors(model, image_dir, n=1000,
                                              save_path='models/hsi/band_selection/bam_scaled_attn.npy'):
    """
    Samples n images from the image directory, processes them through the model's BAM module to
    get the scaled attention vectors, and saves the resulting 2D matrix of size (n, C) to an .npy file.

    Args:
        model: The BandAttentionIntegrated model (with the BAM module loaded).
        image_dir: Directory containing hyperspectral images in .npy format.
        n: Number of images to sample from the directory.
        save_path: File path to save the concatenated scaled attention vectors as .npy.
    """
    # List all .npy files in the directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')]

    # Randomly sample n files
    sampled_files = random.sample(image_files, n)

    attn_vectors = []

    # Process each sampled image
    for file in tqdm(sampled_files, desc = "Getting Scaled Attention Vectors"):
        # Load the .npy image
        data = np.load(file)

        # Convert the image data to tensor and move to GPU (if available)
        input_image_tensor = torch.from_numpy(data).float().permute(2, 0, 1).unsqueeze(0).to('cuda')

        # Get the scaled attention vector
        scaled_attn_vector = get_scaled_attention_vector(model, input_image_tensor).squeeze(2)

        # Add the vector to the list (detaching and moving to CPU)
        attn_vectors.append(scaled_attn_vector.cpu().numpy())

    # Concatenate the list of vectors along dim 0 to create a matrix of size (1000, C)
    attn_matrix = np.concatenate(attn_vectors, axis=0)

    # Save the concatenated matrix as a .npy file
    np.save(save_path, attn_matrix)
    print(f"Saved scaled attention vectors to {save_path}")

def rank_channels_by_attention_score(npy_file):
    """
    Loads the scaled attention vectors from the given .npy file, computes the average score for each channel (column),
    and ranks the channels in decreasing order of their scores.

    Args:
        npy_file: Path to the .npy file containing the attention vectors.

    Returns:
        ranked_channels (list): List of channel indices ranked in decreasing order of their average attention scores.
        avg_scores (list): List of average attention scores corresponding to the ranked channels.
    """
    # Load the scaled attention vectors from the .npy file
    attn_matrix = np.load(npy_file)
    print(attn_matrix.shape)

    # Calculate the mean score for each channel (mean of each column)
    avg_scores = np.mean(attn_matrix, axis=0)

    # Rank the channels by their scores in decreasing order
    ranked_channels = np.argsort(avg_scores)[::-1]

    # Get the sorted scores corresponding to the ranked channels
    ranked_scores = avg_scores[ranked_channels]

    # Return a tuple of ranked channels and their scores
    return list(ranked_channels), list(ranked_scores)

def process_images_by_channel_count(source_dir, dest_dir, channel_ranking, increment=25):
    """
    Processes images from source_dir, selects channels based on the given ranking, and saves
    them in sub-directories of dest_dir with an incremental count of channels.

    Args:
        source_dir: Directory containing the original .npy images.
        dest_dir: Directory where processed images will be saved.
        channel_ranking: List of channel indices ranked by importance.
        increment: The number of channels to increment by each step (default is 25).
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all .npy files in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]

    # Process images for increments of channels
    channel_counts = list(range(increment, len(channel_ranking)+1, increment)) + [len(channel_ranking)]

    for count in channel_counts:
        # Create a sub-directory for each count of channels
        sub_dir = os.path.join(dest_dir, f"channels_{count}")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # Select the top 'count' channels from the ranking
        selected_channels = channel_ranking[:count]

        # Process each image in the source directory
        for image_file in tqdm(image_files, desc = f"Band Selection: {count} Channels"):
            # Load the .npy image (format H, W, C)
            image_path = os.path.join(source_dir, image_file)
            image_data = np.load(image_path)

            # Select the specified channels from the image
            selected_image_data = image_data[:, :, selected_channels]

            # Save the processed image in the sub-directory with the same name
            save_path = os.path.join(sub_dir, image_file)
            np.save(save_path, selected_image_data)

        print(f"Processed and saved images with {count} channels in {sub_dir}")

import pickle
if __name__ == '__main__':
    ckpt = "results/hsi/classes-96/ckpts/sparse_bam_densenet_withoutTemp_LossMean__maskedPCALoading__none__168--epoch=186-val_loss=0.00-val_accuracy=0.85.ckpt"
    suffix = "_withoutTemp_LossMean"
    state_dict = torch.load(ckpt, weights_only=True)['state_dict']
    weights = {k.replace("model_obj.model.", ""): v for k, v in state_dict.items() if k.startswith("model_obj.model.")}
    # print(weights.keys())
    config = read_yaml('models/hsi/config.yaml')
    input = torch.randn(32, config['C'], config['H'], config['W']).to('cuda')
    A = BandAttentionIntegrated(config).to('cuda')
    A.model.load_state_dict(weights)

    process_and_save_scaled_attention_vectors(A, 'Data/hsi_masked/', n=2000,
                                              save_path=f'models/hsi/band_selection/bam_scaled_attn{suffix}.npy')

    ranked_channels, ranked_scores = rank_channels_by_attention_score(f'models/hsi/band_selection/bam_scaled_attn{suffix}.npy')
    pickle.dump({'channels': ranked_channels, 'scores': ranked_scores}, open(f'models/hsi/band_selection/bam_ranked_channels{suffix}.pkl', 'wb'))
    for channel, score in zip(ranked_channels, ranked_scores):
        print(f"Channel {channel}: {score:.4f}")


    process_images_by_channel_count('Data/hsi_masked', f'Data/hsi_masked_bam{suffix}', ranked_channels, increment=25)

