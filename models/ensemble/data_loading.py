import  sys
import pandas as pd
sys.path.append('Preprocessing')
from train_eval import *
from callbacks import *

fold = 0
class_ct = 96
config = {
  'tr_path' : 'Data/{class_ct}/fold_{x}/df_tr.csv'.format(class_ct=class_ct, x = fold),
  'val_path' : 'Data/{class_ct}/fold_{x}/df_val.csv'.format(class_ct=class_ct, x = fold),
  'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
}

#_______________________________________________________________________________________________________________________

class EnsembleDataset(Dataset):
  # defining values in the constructor
  def __init__(self , df,rgb_transforms , hsi_transforms):
    self.df = df
    self.Y = torch.tensor( self.df.iloc[:, -1].values, dtype=torch.float32)
    self.shape = self.df.shape
    self.rgb_transforms = rgb_transforms
    self.hsi_transforms = hsi_transforms
    
  # Getting the data samples
  def __getitem__(self, idx):
    y =  self.Y[idx]
    img_tensor = None
    rgb_img_path = self.df.iloc[idx, 0]
    hsi_img_path = self.df.iloc[idx, 1]

    rgb_img_tensor = Image.open(rgb_img_path)
    hsi_img_tensor = np.load(hsi_img_path)
    hsi_img_tensor = self.__msc_correction__(hsi_img_tensor)
      # img_tensor = self.__snv__(img_tensor)
    
    rgb_img_tensor = self.rgb_transforms(rgb_img_tensor)
    hsi_img_tensor = self.hsi_transforms(hsi_img_tensor)
    return (hsi_img_tensor, rgb_img_tensor), y
  
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
  
#_______________________________________________________________________________________________________________________
df_tr = pd.read_csv(config['tr_path'])
tr_dataset = EnsembleDataset(df_tr, rgb_transforms , hsi_img_transforms)

df_val = pd.read_csv(config['val_path'])
val_dataset = EnsembleDataset(df_val, rgb_transforms, hsi_img_transforms)

df_tst = pd.read_csv(config['tst_path'])
tst_dataset = EnsembleDataset(df_tst, rgb_transforms, hsi_img_transforms)