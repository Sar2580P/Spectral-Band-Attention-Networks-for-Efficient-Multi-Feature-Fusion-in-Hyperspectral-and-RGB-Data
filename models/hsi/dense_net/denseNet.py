import  sys
sys.path.append('models')
sys.path.append('Preprocessing')
sys.path.append('models/hsi')
from train_eval import *
from model import DenseNet
from callbacks import *
from utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from data_loading import *
import pickle

config_path = 'models/hsi/dense_net/config.yaml'
config = load_config(config_path)
torch.set_float32_matmul_precision('high')

model_parameters={}
model_parameters['densenet_mini'] = [12, 18, 24, 6]
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]
model_parameters['densenet264'] = [6,12,64,48]

model_obj = DenseNet(densenet_variant = model_parameters['densenet_mini'] , in_channels=config['in_channels'],
                     num_classes=config['num_classes'] , compression_factor=0.3 , k = 32 , config=config)
model = Classifier(model_obj)
# hsi_dense_net_ckpt = 'models/hsi/dense_net/ckpts/dense_net--epoch=129-val_loss=0.39-val_accuracy=0.87.ckpt'
# model = Classifier.load_from_checkpoint(hsi_dense_net_ckpt, model_obj=model_obj)

num_workers = 8
tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)

#___________________________________________________________________________________________________________________

NAME = config['model_name']+f'__var-{config["num_classes"]}'+f'__fold-{config["fold"]}'

checkpoint_callback.dirpath = os.path.join(config['dir'], 'ckpts')
checkpoint_callback.filename = NAME+'__' + config['ckpt_file_name']


run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
wandb_logger = WandbLogger(project=NAME , name = run_name)
csv_logger = CSVLogger(config['dir']+'/logs/'+ NAME)


trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,max_epochs=300)  # , logger=[wandb_logger,csv_logger]

trainer.fit(model, tr_loader, val_loader)
trainer.test(model, tst_loader)

if not os.path.exists(os.path.join(config['dir'], 'evaluations')):
  os.mkdir(os.path.join(config['dir'], 'evaluations'))


with open(os.path.join(config['dir'], 'evaluations', NAME+'__predictions.pkl'), 'wb') as f:
  dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true}
  pickle.dump(dict_, f)
