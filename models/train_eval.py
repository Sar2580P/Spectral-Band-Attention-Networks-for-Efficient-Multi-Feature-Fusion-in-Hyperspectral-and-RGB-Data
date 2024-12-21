import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
import torch.nn as nn

# https://stackoverflow.com/a/73747249/22707633
class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("Multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0

            self.base_lrs = [initial_lr * (self.decay ** n) for initial_lr in self.initial_lrs]

        super().step(epoch)

# Function to plot and save learning rate
def plot_lr_scheduler(optimizer , eta_max, T_0, T_mult, max_epochs, decay, filename='lr_plot.png'):
    # Set the initial learning rate for the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = eta_max

    # Initialize the custom scheduler
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=T_0, T_mult=T_mult, decay=decay)

    # List to store learning rates for each epoch
    lrs = []

    # Simulate the training loop
    for epoch in range(max_epochs):
        scheduler.step(epoch)
        # Record the learning rate
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot the learning rate over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_epochs), lrs, label=f'CosineAnnealingWarmRestartsDecay\nT_0={T_0}, T_mult={T_mult}, decay={decay}')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

# # Example usage:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# plot_lr_scheduler(eta_max=0.00001, T_0=25, T_mult=2, max_epochs=300, decay=0.8, filename='lr_decay_plot.png')


class BandSparsityLoss(nn.Module):
    def __init__(self, beta: float, total_epochs: int, initial_lambda: float = 1e-4, loss_type='l1'):
        """
        Initializes the auxiliary loss class.

        Parameters:
        beta (float): Coefficient to control how fast lambda decays.
        total_epochs (int): Total number of epochs for training.
        initial_lambda (float): Initial value of lambda (adjustment parameter).
        """
        super(BandSparsityLoss, self).__init__()
        self.beta = beta
        self.total_epochs = total_epochs
        self.lambda_ = initial_lambda
        self.loss_type = loss_type

    def update_lambda(self, current_epoch: int):
        t1 = current_epoch
        t2 = self.total_epochs/5   # limiting the decay of lambda to 1/5th of the total epochs
        self.lambda_ = self.beta * math.exp(-t1 / t2)

    def forward(self, w_ij: torch.Tensor, current_epoch: int):
      """
      Computes the auxiliary loss.

      Parameters:
      w_ij (torch.Tensor): Tensor representing non-sparse constraint weights for each band of each sample.
      current_epoch (int): The current epoch number (passed dynamically).

      Returns:
      torch.Tensor: The computed auxiliary loss.
      """
      # Update lambda based on the current epoch
      self.update_lambda(current_epoch)

      # Calculate the mean weights across the batch axis
      mean_weights = torch.mean(w_ij, dim=0)


      if self.loss_type == 'l1':
        # Calculate L1 norm of the mean weights
        band_sparsity_loss = torch.sum(torch.abs(mean_weights))  # Use sum of absolute values for L1 norm
      elif self.loss_type == 'mean':
        band_sparsity_loss = torch.mean(mean_weights)

      # Return the auxiliary loss scaled by lambda
      return self.lambda_ * band_sparsity_loss

class Classifier(pl.LightningModule):
  def __init__(self, model_obj):
    super().__init__()
    self.model_obj = model_obj
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

    if self.model_obj.model_name.startswith('sparse_bam'):
      self.aux_loss_fn = BandSparsityLoss(beta=self.config['sparse_bam_config']['beta'],
                                          total_epochs=self.config['MAX_EPOCHS'],
                                          loss_type=self.config['sparse_bam_config']['loss_type']
                                          )

    self.y_hat = []
    self.y_true = []


  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss
    total_loss = ce_loss

    if self.model_obj.model_name.startswith('sparse_bam'):
      w_ij = self.model_obj.bam.scaled_attn_vector  # Assuming w_ij is stored in model_obj and is updated each epoch
      # Get the current epoch from the Trainer object
      current_epoch = self.current_epoch
      # Compute auxiliary loss with the current epoch passed as a parameter
      band_sparsity_loss = self.aux_loss_fn(w_ij, current_epoch)
      total_loss += band_sparsity_loss
      self.log("band_sparsity_train_loss", band_sparsity_loss, on_step = True ,on_epoch=True, prog_bar=True, logger=True)
      self.log("mean_weights" , torch.mean(w_ij), on_step = False ,on_epoch=True, prog_bar=True, logger=True)
      self.log("std_weights" , torch.std(w_ij), on_step = False ,on_epoch=True, prog_bar=True, logger=True)
      self.log("temperature" , self.model_obj.bam.temperature, on_step = False ,on_epoch=True, prog_bar=True, logger=True)
      self.log("sparsity_threshold" , self.model_obj.bam.learnable_threshold, on_step = False ,on_epoch=True, prog_bar=True, logger=True)


    self.tr_accuracy(y_hat, y)
    self.tr_kappa(y_hat, y)
    self.log("cross_entropy_train_loss", ce_loss,on_step = True ,on_epoch=True, prog_bar=True, logger=True)
    self.log("total_train_loss", total_loss, on_step = True ,on_epoch=True, prog_bar=True, logger=True)
    self.log("train_kappa", self.tr_kappa, on_step=True , on_epoch=True, prog_bar=True, logger=True)
    self.log("train_accuracy", self.tr_accuracy, on_step = True, on_epoch=True,prog_bar=True, logger=True)

    return total_loss

  def validation_step(self, batch, batch_idx):
    # bring model to same device as batch
    # self.model_obj.model.to(batch[0].device)
    x, y = batch
    y_hat = self.model_obj.forward(x)
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss
    total_loss = ce_loss

    if self.model_obj.model_name.startswith('sparse_bam'):
      w_ij = self.model_obj.bam.scaled_attn_vector  # Assuming w_ij is stored in model_obj and is updated each epoch
      # Get the current epoch from the Trainer object
      current_epoch = self.current_epoch
      # Compute auxiliary loss with the current epoch passed as a parameter
      band_sparsity_loss = self.aux_loss_fn(w_ij, current_epoch)
      self.log("band_sparsity_val_loss", band_sparsity_loss, on_step = False ,on_epoch=True, prog_bar=True, logger=True)

      total_loss += band_sparsity_loss

    self.val_accuracy(y_hat, y)
    self.val_kappa(y_hat, y)
    self.log("cross_entropy_val_loss", ce_loss, on_epoch=True, prog_bar=True, logger=True)
    self.log("total_val_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_kappa", self.val_kappa,on_step = False, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_accuracy", self.val_accuracy, on_step = False, on_epoch=True,prog_bar=True, logger=True)
    return total_loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model_obj.forward(x)
    self.y_hat.append(y_hat)
    self.y_true.append(y)
    ce_loss = self.criterion(y_hat, y.long())      # compute CE loss
    total_loss = ce_loss

    if self.model_obj.model_name.startswith('sparse_bam'):
      w_ij = self.model_obj.bam.scaled_attn_vector  # Assuming w_ij is stored in model_obj and is updated each epoch
      # Get the current epoch from the Trainer object
      current_epoch = self.current_epoch
      # Compute auxiliary loss with the current epoch passed as a parameter
      band_sparsity_loss = self.aux_loss_fn(w_ij, current_epoch)
      self.log("band_sparsity_test_loss", band_sparsity_loss, on_step = False ,on_epoch=True, prog_bar=True, logger=True)

      total_loss += band_sparsity_loss

    self.tst_accuracy(y_hat, y)
    self.tst_kappa(y_hat, y)
    self.log("cross_entropy_test_loss", ce_loss,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("total_test_loss", total_loss,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_kappa", self.tst_kappa,on_step = True,  on_epoch=True, prog_bar=True, logger=True)
    self.log("test_accuracy", self.tst_accuracy, on_step = True ,on_epoch=True,prog_bar=True, logger=True)
    return total_loss

  def configure_optimizers(self):
    optim =  torch.optim.Adam(self.layer_lr, lr = self.config['lr'], weight_decay = self.config['weight_decay'])   # https://pytorch.org/docs/stable/optim.html
    scheduler_params = self.config[f"{self.config['scheduler_name']}_params"]

    if self.config['scheduler_name'] == 'cosine_warm_restarts_decay_lr_scheduler':
      lr_scheduler = CosineAnnealingWarmRestartsDecay(optim, **scheduler_params)

    elif self.config['scheduler_name'] == 'exponential_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)

    elif self.config['scheduler_name'] == 'cosine_decay_lr_scheduler':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)

    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_loss', 'name': self.config['scheduler_name']}]

#___________________________________________________________________________________________________________________
class MyDataset(Dataset):
  # defining values in the constructor
  def __init__(self , df, data_dir:str , hsi_preprocessing:str =None , transforms_ = None , ):
    self.df = df
    self.df.reset_index(drop=True, inplace=True)
    self.Y = torch.tensor( self.df.loc[:, 'class_id'].values, dtype=torch.float32)
    assert len(self.df) == len(self.Y) , "Length of the dataframe and the target tensor should be the same"
    self.shape = self.df.shape
    self.transforms_ = transforms_
    self.hsi_preprocessing = hsi_preprocessing
    self.data_dir = data_dir

  # Getting the data samples
  def __getitem__(self, idx):
    y =  self.Y[idx]
    img_tensor = None
    img_path = os.path.join(self.data_dir , self.df.img_path.iloc[idx])

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
    mean_spectrum = np.mean(hsi_image, axis=(0, 1))

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
      return len(self.df)

