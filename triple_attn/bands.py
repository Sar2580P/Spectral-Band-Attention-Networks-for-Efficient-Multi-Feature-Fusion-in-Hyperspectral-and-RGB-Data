import pickle
import torch
import numpy as np
import skimage.measure
from models.hsi.band_selection.triple_attention import BandAttentionIntegrated
from processing.utils import read_yaml
from torch.utils.data import DataLoader
from models.hsi.data_loading import get_hsi_loaders

if __name__ == '__main__':
    # Load model checkpoint
    ckpt = "results/hsi/classes-96/ckpts/triple_attention__TripleAttnDataset__none__168--epoch=196-val_loss=0.00-val_accuracy=0.85.ckpt"
    state_dict = torch.load(ckpt, weights_only=True)['state_dict']
    weights = {k.replace("model_obj.model.", ""): v for k, v in state_dict.items() if k.startswith("model_obj.model.")}


    # Load model configuration
    config = read_yaml('models/hsi/config.yaml')
    A = BandAttentionIntegrated(config).to('cuda')
    A.model.load_state_dict(weights)
    A.eval()  # Set model to evaluation mode
    triple_attn = A.bam  # Triple Attention Module


    # Function to compute entropy for each spectral band
    def compute_entropy(tensor):
        tensor_np = tensor.detach().cpu().numpy()
        entropy_values = np.array([
            skimage.measure.shannon_entropy(tensor_np[:, i, :, :]) for i in range(tensor.shape[1])
        ])
        return entropy_values

    # Load dataset
    tr_loader, val_loader, tst_loader = get_hsi_loaders(config['data_config'])

    entropy_list = []

    # Extract entropy-based top bands
    with torch.no_grad():
        for img, _ in tr_loader:
            img = img.to('cuda')
            band_features = triple_attn(img)    # Extract features
            entropy_values = compute_entropy(band_features)
            entropy_list.append(entropy_values)

    # Average entropy over all images
    entropy_values = np.mean(entropy_list, axis=0)

    # Select top bands based on entropy and sort them in descending order
    band_counts = [25, 50, 75, 100, 125, 150]
    sorted_indices = np.argsort(entropy_values)[::-1]  # Sort in descending order
    selected_bands = {count: sorted_indices[:count] for count in band_counts}

    # Print top bands in sorted order
    for count, bands in selected_bands.items():
        print(f"Top {count} bands (sorted):", bands.tolist())

    # Save selected bands in sorted order
    with open("selected_bands.pkl", "wb") as f:
        pickle.dump(selected_bands, f)
