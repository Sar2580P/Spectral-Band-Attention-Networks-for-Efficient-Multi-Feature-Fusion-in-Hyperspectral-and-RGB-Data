import  sys
from token import NAME
sys.path.append('models')
sys.path.append('Preprocessing')
sys.path.append('models/rgb')
from train_eval import *
from model import RGB_Resnet
from callbacks import *
from utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from data_loading import *
import pickle

config_path = 'models/rgb/resnet-50/config.yaml'
config = load_config(config_path)
torch.set_float32_matmul_precision('high')

model_obj = RGB_Resnet(config)

#___________________________________________________________________________________________________________________
num_workers = 8
tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers )
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)

#___________________________________________________________________________________________________________________
# rgb_gnet_ckpt = 'models/rgb/google_net/ckpts/gnet--epoch=2-val_loss=1.99-val_accuracy=0.44.ckpt'
# ckpt = torch.load(rgb_gnet_ckpt)
# print(ckpt['state_dict'].keys())
# model = Classifier.load_from_checkpoint(rgb_gnet_ckpt, model_obj=model_obj)
model = Classifier(model_obj)

NAME = config['model_name']+f'__var-{config["num_classes"]}'+f'__fold-{config["fold"]}'
checkpoint_callback.dirpath = os.path.join(config['dir'], 'ckpts')
checkpoint_callback.filename = NAME+'__' + config['ckpt_file_name']

run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
wandb_logger = WandbLogger(project= NAME, name = run_name)
csv_logger = CSVLogger(config['dir']+'/logs/'+ NAME)

trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,max_epochs=65)  # , logger=[wandb_logger, csv_logger]

trainer.fit(model, tr_loader, val_loader)
trainer.test(model, tst_loader)

if not os.path.exists(os.path.join(config['dir'], 'evaluations')):
  os.mkdir(os.path.join(config['dir'], 'evaluations'))


with open(os.path.join(config['dir'], 'evaluations', NAME+'__predictions.pkl'), 'wb') as f:
  dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true}
  pickle.dump(dict_, f)


