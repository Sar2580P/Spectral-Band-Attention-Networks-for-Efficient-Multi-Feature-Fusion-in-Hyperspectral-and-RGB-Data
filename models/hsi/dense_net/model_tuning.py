import  sys
sys.path.append('models')
sys.path.append('Preprocessing')
sys.path.append('models/hsi')
from train_eval import *
from model import DenseNet
from callbacks import *
from utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from data_loading import *
import wandb
from pytorch_lightning.accelerators import find_usable_cuda_devices

sweep_config_path = 'models/hsi/dense_net/architecture_sweep_config.yaml'
sweep_config = load_config(sweep_config_path)

wandb.login()

NAME = sweep_config['parameters']['model_name']['value']+f"__var-{sweep_config['parameters']['num_classes']['value']}"+ \
    f"__fold-{sweep_config['parameters']['fold']['value']}"

print('NAME : ' , NAME , '\n\n')
sweep_id = wandb.sweep(sweep_config, project=NAME)

def tune_hyperparams(config = None):
    with wandb.init(config = config):
        config = wandb.config
        print(config,'\n\n\n')
        num_workers = 8
        tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)

        model_obj = DenseNet(densenet_variant = config['model_size'] , in_channels=config['in_channels'],
                     num_classes=config['num_classes'] , compression_factor=0.3 , k = 32 , config=config)
        model = Classifier(model_obj)

        run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
        wandb_logger = WandbLogger(project=NAME , name = run_name)


        trainer = Trainer(callbacks=[rich_progress_bar],
                        accelerator = 'gpu' ,max_epochs=config['epochs'], logger=[wandb_logger] , devices=find_usable_cuda_devices(1))

        trainer.fit(model, tr_loader, val_loader)
    wandb.finish()


wandb.agent(sweep_id, function=tune_hyperparams, count=40)