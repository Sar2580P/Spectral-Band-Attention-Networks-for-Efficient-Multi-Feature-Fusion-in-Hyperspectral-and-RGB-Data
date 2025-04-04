
import pickle
import torch
from models.hsi.band_selection.triple_attention import BandAttentionIntegrated
from processing.utils import read_yaml
from models.hsi.data_loading import get_hsi_loaders
import os
from tqdm import tqdm
import numpy as np

def create_dataset_for_channels(channel_ranking_path, source_dir, dest_dir):
    """
    Create a dataset for the given channel ranking path.
    """
    with open(channel_ranking_path, 'rb') as f:
        data = pickle.load(f)

    image_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]

    for channel_ct, channel_rankings in data.items():
        sub_dir = os.path.join(dest_dir, f"channels_{channel_ct}")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # Process each image in the source directory
        for image_file in tqdm(image_files, desc = f"Band Selection: {channel_ct} Channels"):
            # Load the .npy image (format H, W, C)
            image_path = os.path.join(source_dir, image_file)
            image_data = np.load(image_path)

            # Select the specified channels from the image
            selected_image_data = image_data[:, :, channel_rankings]

            # Save the processed image in the sub-directory with the same name
            save_path = os.path.join(sub_dir, image_file)
            np.save(save_path, selected_image_data)

        print(f"Processed and saved images with {channel_ct} channels in {sub_dir}")


if __name__ == '__main__':
    # ckpt = "results/hsi/classes-96/ckpts/triple_attention__TripleAttnDataset__none__168--epoch=196-val_loss=0.00-val_accuracy=0.85.ckpt"

    # state_dict = torch.load(ckpt, weights_only=True)['state_dict']
    # weights = {k.replace("model_obj.model.", ""): v for k, v in state_dict.items() if k.startswith("model_obj.model.")}
    # # print(weights.keys())
    # config = read_yaml('models/hsi/config.yaml')

    # A = BandAttentionIntegrated(config).to('cuda')
    # A.model.load_state_dict(weights)

    # triple_attn = A.bam   #[25, 50 ,75, 100, 125, 150, 168]
    # data_dir = 'Data/hsi'
    # tr_loader, val_loader, tst_loader = get_hsi_loaders(config['data_config'])
    create_dataset_for_channels('shitiz/selected_bands.pkl', 'Data/hsi', f'Data/hsi_triple_attn')
































    # process_and_save_scaled_attention_vectors(A, 'Data/hsi/', n=2000,
    #                                           save_path=f'models/hsi/band_selection/triple_attention.npy')

    # ranked_channels, ranked_scores = rank_channels_by_attention_score(f'models/hsi/band_selection/triple_attention.npy')
    # pickle.dump({'channels': ranked_channels, 'scores': ranked_scores}, open(f'models/hsi/band_selection/triple_attention_ranked_channels.pkl', 'wb'))
    # for channel, score in zip(ranked_channels, ranked_scores):
    #     print(f"Channel {channel}: {score:.4f}")


    # process_images_by_channel_count('Data/hsi', f'Data/hsi_triple_attention', ranked_channels, increment=25)

