from rebalta.train_attention import load_model
from processing.utils import read_yaml
import torch 
import torchmetrics


class BandSelection_TrainLoop:
  def __init__(self, model_obj, config):
    super().__init__()
    self.model_obj = model_obj
    self.config = config

    self.tr_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['num_classes'], weights = 'quadratic')
    self.val_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['num_classes'], weights = 'quadratic')
    self.tst_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['num_classes'], weights = 'quadratic')

    self.tr_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])
    self.val_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])
    self.tst_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])

    self.criterion = torch.nn.CrossEntropyLoss()
    self.y_hat = []
    self.y_true = []


  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj(x, y, infer=False)
    loss = self.criterion(y_hat, y.long())      # compute CE loss
    self.tr_accuracy(y_hat, y)
    self.tr_kappa(y_hat, y)
    self.log("train_loss", loss, on_step = True ,on_epoch=True, prog_bar=True, logger=True)
    self.log("train_kappa", self.tr_kappa, on_step=True , on_epoch=True, prog_bar=True, logger=True)
    self.log("train_accuracy", self.tr_accuracy, on_step = True, on_epoch=True,prog_bar=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj(x, y, infer=False)
    loss = self.criterion(y_hat, y.long())      # compute CE loss
    self.val_accuracy(y_hat, y)
    self.val_kappa(y_hat, y)
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_kappa", self.val_kappa,on_step = False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_accuracy", self.val_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj(x, y, infer=False)
    loss = self.criterion(y_hat, y.long())      # compute CE loss
    self.tst_accuracy(y_hat, y)
    self.tst_kappa(y_hat, y)
    self.log("test_loss", loss ,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_kappa", self.tst_kappa,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_accuracy", self.tst_accuracy, on_step = True ,on_epoch=True,prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    optim =  torch.optim.Adam(params=self.model_obj.parameters() , lr = self.config['lr'], weight_decay = self.config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    scheduler_params = self.config[f"{self.config['scheduler_name']}_params"]

    if self.config['scheduler_name'] == 'exponential_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)

    elif self.config['scheduler_name'] == 'cosine_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_loss', 'name': self.config['scheduler_name']}]



if __name__ == "__main__":
    from models.hsi.data_loading import get_hsi_loaders
    import os 
    from models.callbacks import (early_stop_callback, checkpoint_callback, 
                              rich_progress_bar, rich_model_summary, lr_monitor)
    import pickle
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger, CSVLogger
    
    config = read_yaml('rebalta/config.yaml')

    try:
        model_obj = load_model(config['model_config'])
    except Exception as e:
        print(f"Error loading model: {e}")
        raise ValueError("Model loading failed. Please check the model configuration.")
    model = BandSelection_TrainLoop(model_obj, config)
    tr_loader, val_loader, tst_loader = get_hsi_loaders(config['data_config'])
    
    #__________________________________________________________________________________________________________
    NAME = "rebalta_hsi_band_selection"
    RESULT_DIR = os.path.join(config['dir'], f"classes-{config['num_classes']}")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    file_name = NAME+f"__{config['preprocessing']}"+f"__{config['C']}"
    checkpoint_callback.dirpath = os.path.join(RESULT_DIR, 'ckpts')
    checkpoint_callback.filename = file_name +'--'+ config['ckpt_file_name']
    #__________________________________________________________________________________________________________
    
    run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
    wandb_logger = WandbLogger(project= NAME, name = run_name)
    csv_logger = CSVLogger(RESULT_DIR+'/logs/'+ file_name)

    torch.set_float32_matmul_precision('high')
    trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                        accelerator = 'gpu' ,accumulate_grad_batches=config['accumulate_grad_batches'] , max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger])

    trainer.fit(model, tr_loader, val_loader)
    trainer.test(model, tst_loader)

    if not os.path.exists(os.path.join(RESULT_DIR, 'evaluations')):
        os.mkdir(os.path.join(RESULT_DIR, 'evaluations'))


        with open(os.path.join(RESULT_DIR, 'evaluations', f"{file_name}__predictions.pkl"), 'wb') as f:
            dict_ = {'y_hat': model.y_hat, 'y_true': model.y_true , 'training_config' : config}
            pickle.dump(dict_, f)

    else:
        print("Model already trained, with file_name: ", file_name)