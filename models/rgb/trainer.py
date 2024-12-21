from models.train_eval import Classifier
from torch.utils.data import DataLoader
from models.model_architectures import RGB_Resnet , DenseNetRGB , GoogleNet
from models.callbacks import early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary
from processing.utils import read_yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from models.rgb.data_loading import tr_dataset, val_dataset, tst_dataset
import pickle
import torch
import os

torch.set_float32_matmul_precision('high')
config = read_yaml('models/rgb/config.yaml')

get_model = {
    'resnet' : RGB_Resnet,
    'densenet' : DenseNetRGB,
    'google_net' : GoogleNet,
}

#___________________________________________________________________________________________________________________
tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])

#___________________________________________________________________________________________________________________
# rgb_gnet_ckpt = 'models/rgb/google_net/ckpts/gnet--epoch=2-val_loss=1.99-val_accuracy=0.44.ckpt'
# ckpt = torch.load(rgb_gnet_ckpt)
# print(ckpt['state_dict'].keys())
# model = Classifier.load_from_checkpoint(rgb_gnet_ckpt, model_obj=model_obj)
model_obj = get_model[config['model_name']](config)
model = Classifier(model_obj)

NAME = model_obj.model_name+f'__var-{config["num_classes"]}'
RESULT_DIR = os.path.join(config['dir'], f"classes-{config['num_classes']}")

if not os.path.exists(RESULT_DIR):
  os.makedirs(RESULT_DIR)

checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
checkpoint_callback.filename = model_obj.model_name +'--'+ config['ckpt_file_name']

run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
wandb_logger = WandbLogger(project= NAME, name = run_name)
csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ model_obj.model_name)

trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger],
                  accumulate_grad_batches=config['accumulate_grad_batches'])

trainer.fit(model, tr_loader, val_loader)
trainer.test(model, dataloaders= tst_loader)

if not os.path.exists(os.path.join(RESULT_DIR, 'evaluations')):
  os.mkdir(os.path.join(RESULT_DIR, 'evaluations'))


with open(os.path.join(RESULT_DIR, 'evaluations', model_obj.model_name+'_predictions.pkl'), 'wb') as f:
  dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true , 'training_config' : config}
  pickle.dump(dict_, f)


