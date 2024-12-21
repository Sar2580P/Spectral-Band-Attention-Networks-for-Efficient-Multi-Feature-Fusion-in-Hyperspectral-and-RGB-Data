from models.train_eval import Classifier
from torch.utils.data import DataLoader
from models.model_architectures import DenseNet
from models.hsi.band_selection.bam import BandAttentionIntegrated
from models.callbacks import rich_progress_bar, rich_model_summary
from processing.utils import read_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from models.hsi.data_loading import tr_dataset, val_dataset, tst_dataset
import os
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

sweep_config =read_yaml('models/hsi/dense_net/sweep_config.yaml')
es_config = read_yaml('models/hsi/config.yaml')['EarlyStopping']
# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='hsi_densenet_project')

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        if config['model_name'] == 'densenet':
            model_obj = DenseNet(config=config, plot_model_arch=False)

        elif config['model_name'] == 'sparse_bam_densenet':
            config['sparse_bam_config'] = {
                'r' : config['r'],
                'beta' : config['beta'],
                'head_model_name' : config['head_model_name'],
                'head_model_ckpt' : config['head_model_ckpt'],
                'sparsity_threshold' : config['sparsity_threshold'],
                'temperature' : config['temperature']

            }
            config =  dict(config)
            model_obj = BandAttentionIntegrated(config)

        model = Classifier(model_obj)

        tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])
        tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])

        NAME = model_obj.model_name + f'__var-{config["num_classes"]}'
        RESULT_DIR = os.path.join(config['dir'], f"classes-{config['num_classes']}")

        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        file_name = model_obj.model_name + f"__{config['data_type']}" + f"__{config['preprocessing']}"


        run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
        wandb_logger = WandbLogger(project=NAME, name=run_name)


        early_stop_callback = EarlyStopping(
            monitor=es_config['monitor'],
            min_delta=es_config['min_delta'],
            patience=es_config['patience'],
            verbose=es_config['verbose'],
            mode=es_config['mode']
        )
        torch.set_float32_matmul_precision('high')
        trainer = Trainer(callbacks=[early_stop_callback, rich_progress_bar, rich_model_summary],
                          accelerator='gpu', max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger] ,
                          accumulate_grad_batches=config['accumulate_grad_batches'])

        trainer.fit(model, tr_loader, val_loader)
        trainer.test(model, tst_loader)

# Run the sweep
wandb.agent(sweep_id, function=train, count = 50)