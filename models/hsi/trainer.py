from models.train_eval import Classifier
from models.model_architectures import DenseNet
from models.callbacks import (early_stop_callback, checkpoint_callback, 
                              rich_progress_bar, rich_model_summary, lr_monitor)
from processing.utils import read_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from data_loading import get_hsi_loaders
import pickle
import torch
import os
from models.hsi.band_selection.bam import BandAttentionIntegrated
import glob

config = read_yaml('models/hsi/config.yaml')

if config['model_name'].startswith('densenet'):
  model_obj = DenseNet(config=config)
elif config['model_name'].startswith('sparse_bam_densenet'):
  model_obj = BandAttentionIntegrated(config)

if config['load_model'] :
  try :
    # print(f"{config['dir']}/classes-{config['num_classes']}/ckpts/densenet-24-18-16-10__{config['data_type']}__{config['preprocessing']}__{config['C']}--*.ckpt")
    ckpt_path = glob.glob(f"{config['dir']}/classes-{config['num_classes']}/ckpts/densenet-24-18-16-10__{config['data_type']}__{config['preprocessing']}__{config['C']}--*.ckpt")[0]
    model = Classifier.load_from_checkpoint(checkpoint_path=ckpt_path,
                                            model_obj=model_obj)
    print("Model ckpt loaded successfully")
  except Exception as e:
    print(e)
    print("Model ckpt loading failed, creating new model")
    model = Classifier(model_obj)
else:
  model = Classifier(model_obj)

tr_loader, val_loader, tst_loader = get_hsi_loaders(config['data_config'])
#___________________________________________________________________________________________________________________

NAME = model_obj.model_name+f'__var-{config["num_classes"]}'
RESULT_DIR = os.path.join(config['dir'], f"classes-{config['num_classes']}")

if not os.path.exists(RESULT_DIR):
  os.makedirs(RESULT_DIR)

file_name = model_obj.model_name+f"__{config['data_type']}"+f"__{config['preprocessing']}"+f"__{config['C']}"
checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = file_name +'--'+ config['ckpt_file_name']

if  os.path.exists(os.path.join(RESULT_DIR, 'evaluations', file_name+"__predictions.pkl")):
  run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
  wandb_logger = WandbLogger(project= NAME, name = run_name)
  csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ file_name)

  torch.set_float32_matmul_precision('high')
  trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                    accelerator = 'gpu' ,accumulate_grad_batches=config['accumulate_grad_batches'] , max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

  # trainer.fit(model, tr_loader, val_loader)
  trainer.test(model, tst_loader)

  if not os.path.exists(os.path.join(RESULT_DIR, 'evaluations')):
    os.mkdir(os.path.join(RESULT_DIR, 'evaluations'))


  with open(os.path.join(RESULT_DIR, 'evaluations', f"{file_name}__predictions.pkl"), 'wb') as f:
    dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true , 'training_config' : config}
    pickle.dump(dict_, f)

else:
  print("Model already trained, with file_name: ", file_name)

