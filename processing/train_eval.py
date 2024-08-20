import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
import os
import pandas as pd

class Classifier(pl.LightningModule):
  def __init__(self, model_obj):
    super().__init__()
    self.model_obj = model_obj
    self.model = model_obj.model
    self.config = model_obj.config
    self.layer_lr = model_obj.layer_lr

    self.tr_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['num_classes'], weights = 'quadratic')
    self.val_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['num_classes'], weights = 'quadratic')
    self.tst_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['num_classes'], weights = 'quadratic')

    self.tr_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])
    self.val_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])
    self.tst_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])

    self.conf_mat = MulticlassConfusionMatrix(num_classes = self.config['num_classes'])
    self.criterion = torch.nn.CrossEntropyLoss()

    self.y_hat = []
    self.y_true = []


  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    loss = self.criterion(y_hat, y.long())
    self.log("train_loss", loss,on_step = False ,on_epoch=True, prog_bar=True, logger=True)
    self.tr_accuracy(y_hat, y)
    self.tr_kappa(y_hat, y)
    self.log("train_kappa", self.tr_kappa, on_step=False , on_epoch=True, prog_bar=True, logger=True)
    self.log("train_accuracy", self.tr_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    loss = self.criterion(y_hat, y.long())
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    self.val_accuracy(y_hat, y)
    self.val_kappa(y_hat, y)
    self.log("val_kappa", self.val_kappa,on_step = False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_accuracy", self.val_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    self.y_hat.append(y_hat)
    self.y_true.append(y)
    loss = self.criterion(y_hat, y.long())
    self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    self.tst_accuracy(y_hat, y)
    self.tst_kappa(y_hat, y)
    self.log("test_kappa", self.tst_kappa,on_step = False,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_accuracy", self.tst_accuracy, on_step = False ,on_epoch=True,prog_bar=True, logger=True)
    return loss

  # def on_test_epoch_end(self):
  #   y_hat = torch.cat(self.y_hat, dim=0)
  #   y_true = torch.cat(self.y_true, dim=0)

  #   cm = self.conf_mat(y_hat, y_true).detach().cpu().numpy()

  #   df = pd.DataFrame(cm , columns = [str(i) for i in range(self.config['num_classes'])])
  #   df.to_csv(os.path.join(self.config['dir'], 'confusion_matrix.csv'))


  def configure_optimizers(self):
    optim =  torch.optim.Adam(self.layer_lr, lr = self.config['lr'], weight_decay = self.config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.7, threshold=0.005, cooldown =2,verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,gamma = 0.995 ,last_epoch=-1,   verbose=True)

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_loss', 'name': 'lr_scheduler'}]

#___________________________________________________________________________________________________________________
class MyDataset(Dataset):
  # defining values in the constructor
  def __init__(self , df, hsi_preprocessing:str =None , transforms_ = None ):
    self.df = df
    self.Y = torch.tensor( self.df.loc[:, 'class_id'].values, dtype=torch.float32)
    self.shape = self.df.shape
    self.transforms_ = transforms_
    self.hsi_preprocessing = hsi_preprocessing

  # Getting the data samples
  def __getitem__(self, idx):
    y =  self.Y[idx]
    img_tensor = None
    img_path = self.df.img_path.iloc[idx]
    if img_path.lower().split('.')[-1] == 'png':
      img_tensor = Image.open(img_path)
    else :
      img_tensor = np.load(img_path)
      if self.hsi_preprocessing == 'snv':
        img_tensor = self.__snv__(img_tensor)
      elif self.hsi_preprocessing == 'msc':
        img_tensor = self.__msc_correction__(img_tensor)

    if self.transforms_ is not None:
      img_tensor = self.transforms_(img_tensor)
    return img_tensor, y

  def __msc_correction__(self, hsi_image):

    # Compute the
    mean_spectrum = np.mean(hsi_image, axis=(0, 1))     # mean spectrum of the image , shape = (num_bands,)

    # Compute the MSC correction factor for each band.
    msc_correction_factor = mean_spectrum / np.mean(mean_spectrum)

    # Apply the MSC correction to the image.
    corrected_image = hsi_image * msc_correction_factor

    return corrected_image

  def __snv__(self, data):
    # Compute the mean and standard deviation of each row of the data.
    row_means = np.mean(data, axis=1,  keepdims=True)
    row_stds = np.std(data, axis=1,  keepdims=True)

    # Center and scale each row of the data.
    snv_data = (data - row_means) / row_stds

    return snv_data

  def __len__(self):
    return self.shape[0]
