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



# -- Model Implementation --#

class BasicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1,groups=1,relu=True,bn=True,bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,
                                padding=padding,dilation=dilation,groups=groups,bias=bias)

        self.bn = (nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:     x = self.bn(x)
        if self.relu:   x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1)//2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.hc = AttentionGate()
        self.cw = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()        # Height and Channels
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        x_perm1 = x.permute(0, 2, 1, 3).contiguous()        # Channels and Width
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        if not self.no_spatial:                             # Spatial Attention
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:   x_out = 1/2 * (x_out11 + x_out21)
        return x_out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(EncoderBlock, self).__init__()
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=(3//2)),
                                    nn.PReLU())
        self.conv5x5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (5, 5), stride=1, padding=(5//2)),
                                    nn.PReLU())
        self.maxpool = nn.MaxPool2d((3,3), stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x12 = torch.cat([x1, x2], dim=1)
        out = self.maxpool(x12)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=(3//2)),
                                    nn.PReLU())

    def forward(self, x):
        x = self.upscale(x)
        x = self.conv3x3(x)
        return x

class MSRecNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(MSRecNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, base_channels)
        self.encoder2 = EncoderBlock(base_channels * 2, base_channels * 2)
        self.decoder1 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.decoder2 = DecoderBlock(base_channels * 2, in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        dec1 = self.decoder1(enc2)
        dec2 = self.decoder2(dec1)
        out = self.final_conv(dec2)
        return out

class TattMSRecNet(nn.Module):
    def __init__(self, input_channels):
        super(TattMSRecNet,self).__init__()
        self.triple_attention = TripletAttention()
        # Base channels can be reduced for MSRecNet for decreasing no. of parameters
        self.msrecnet = MSRecNet(in_channels=input_channels, base_channels=input_channels)

    def forward(self,x):
        x = self.triple_attention(x)
        out = self.msrecnet(x)
        return out



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
        self.bam = TattMSRecNet(input_channels=self.config['C'])


        dup_config = self.config.copy()
        dup_config['model_name'] = self.config['sparse_bam_config']['head_model_name']   #the self.config['model_name'] is the name of the BAM integrated model
        if self.config['sparse_bam_config']['head_model_name'] == 'densenet':
            self.head = DenseNet(config=dup_config)


        # self.load_head_ckpt()

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

