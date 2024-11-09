import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from processing.utils import read_yaml
from models.model_architectures import DeepEnsembleModel
from models.train_eval import Classifier
from models.callbacks import (early_stop_callback, checkpoint_callback,
                              rich_progress_bar, rich_model_summary)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import pickle
import glob

hsi_config = read_yaml('models/hsi/config.yaml')

class EnsembleDataSet(Dataset):
    def __init__(self, mode, directory, hsi_config):
        self.mode = mode
        self.directory = directory
        self.tr_files = glob.glob(os.path.join(self.directory, 'rgb_*_train.npy'))
        pattern = f"hsi_*_train_{hsi_config['data_type']}__{hsi_config['preprocessing']}__{hsi_config['C']}.npy"
        self.tr_files += glob.glob(os.path.join(self.directory, pattern))

        self.val_files = [file.replace('train', 'val') for file in self.tr_files]
        self.files = self.tr_files if mode == 'train' else self.val_files

        # Load the first file to get the initial labels
        npy_data = np.load(self.files[0])
        self.data = npy_data[:, :-1]  # All columns except last
        self.labels = npy_data[:, -1]   # Last column

        # Verify all files have the same labels
        for file in self.files[1:]:
            npy_data = np.load(file)
            current_labels = npy_data[:, -1]
            assert np.all(self.labels == current_labels), "Labels across files are not the same."
            self.data = np.concatenate([self.data, npy_data[:, :-1]], axis=1)

        self.labels = self.labels  # Keep the same labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.int8)
        return x, y

def create_dataloader(mode, directory, batch_size=32, shuffle=True):
    dataset = EnsembleDataSet(mode, directory, hsi_config=hsi_config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


config = read_yaml('models/ensemble/config.yaml')['deep_ensemble']
tr_loader = create_dataloader('train', f"results/ensemble/base_models/classes-{config['num_classes']}/fold_{config['fold']}")
val_loader = create_dataloader('val', f"results/ensemble/base_models/classes-{config['num_classes']}/fold_{config['fold']}", shuffle=False)
tst_loader = create_dataloader('test', f"results/ensemble/base_models/classes-{config['num_classes']}/fold_{config['fold']}", shuffle=False)
model_obj = DeepEnsembleModel(config)
model = Classifier(model_obj=model_obj)

#___________________________________________________________________________________________________________________

NAME = model_obj.model_name+f'__var-{hsi_config["num_classes"]}'+f'__fold-{hsi_config["fold"]}'
RESULT_DIR = os.path.join(config['dir'], f"classes-{hsi_config['num_classes']}" , f"fold-{hsi_config['fold']}")

if not os.path.exists(RESULT_DIR):
  os.makedirs(RESULT_DIR)

file_name = model_obj.model_name+f"__{hsi_config['data_type']}"+f"__{hsi_config['preprocessing']}"+f"__{hsi_config['C']}"
checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = file_name + f"--{config['ckpt_file_name']}"

run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
wandb_logger = WandbLogger(project= NAME, name = run_name)
csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ file_name)

torch.set_float32_matmul_precision('high')
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,accumulate_grad_batches=config['accumulate_grad_batches'] ,
                  max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

trainer.fit(model, tr_loader, val_loader)
trainer.test(model, tst_loader)

if not os.path.exists(os.path.join(RESULT_DIR, 'evaluations')):
  os.mkdir(os.path.join(RESULT_DIR, 'evaluations'))


with open(os.path.join(RESULT_DIR, 'evaluations',f"{file_name}__predictions.pkl"), 'wb') as f:
  dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true , 'training_config' : config}
  pickle.dump(dict_, f)