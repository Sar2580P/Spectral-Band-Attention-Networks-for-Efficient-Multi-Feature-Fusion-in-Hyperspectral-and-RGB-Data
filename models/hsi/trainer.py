from models.train_eval import Classifier
from torch.utils.data import DataLoader
from models.model_architectures import DenseNet
from models.callbacks import early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary
from processing.utils import read_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from data_loading import tr_dataset, val_dataset, tst_dataset
import pickle
import torch
import os

config = read_yaml('models/hsi/config.yaml')
torch.set_float32_matmul_precision('high')

densenet_model_parameters={
    'densenet_mini' : [12, 18, 24, 6],
    'densenet121' : [6,12,24,16],
    'densenet169' : [6,12,32,32],
    'densenet201' : [6,12,48,32],
    'densenet264' : [6,12,64,48]
}

if config['model_name'] == 'densenet':
    model_obj = DenseNet(densenet_variant = densenet_model_parameters[config['densenet_variant']] ,config=config)

model = Classifier(model_obj)
# hsi_dense_net_ckpt = 'models/hsi/dense_net/ckpts/dense_net--epoch=129-val_loss=0.39-val_accuracy=0.87.ckpt'
# model = Classifier.load_from_checkpoint(hsi_dense_net_ckpt, model_obj=model_obj)

tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])

#___________________________________________________________________________________________________________________

NAME = model_obj.model_name+f'__var-{config["num_classes"]}'+f'__fold-{config["fold"]}'
RESULT_DIR = os.path.join(config['dir'], f"classes-{config['num_classes']}" , f"fold-{config['fold']}")

if not os.path.exists(RESULT_DIR):
  os.makedirs(RESULT_DIR)

file_name = model_obj.model_name+f"__{config['data_type']}"+f"__{config['preprocessing']}"+('__BAM' if config['apply_BAM'] else '')
checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = file_name +'--'+ config['ckpt_file_name']

run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
wandb_logger = WandbLogger(project= NAME, name = run_name)
csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ file_name)


trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,accumulate_grad_batches=config['accumulate_grad_batches'] , max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

trainer.fit(model, tr_loader, val_loader)
trainer.test(model, tst_loader)

if not os.path.exists(os.path.join(RESULT_DIR, 'evaluations')):
  os.mkdir(os.path.join(RESULT_DIR, 'evaluations'))


with open(os.path.join(RESULT_DIR, 'evaluations', f"{file_name}__predictions.pkl"), 'wb') as f:
  dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true , 'training_config' : config}
  pickle.dump(dict_, f)
