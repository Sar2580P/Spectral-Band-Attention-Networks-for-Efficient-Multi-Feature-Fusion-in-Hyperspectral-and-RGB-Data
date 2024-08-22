import  sys
import pandas as pd
sys.path.append('Preprocessing')
from train_eval import *
from callbacks import *
from utils import *

training_config=load_config('models/hsi/dense_net/config.yaml')

fold = training_config['fold']
class_ct = training_config['num_classes']
config = {
  'tr_path'  : 'Data/{class_ct}/fold_{x}/df_tr.csv'.format(class_ct=class_ct, x = fold),
  'val_path' : 'Data/{class_ct}/fold_{x}/df_val.csv'.format(class_ct=class_ct, x = fold),
  # 'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
}


df_tr = pd.read_csv(config['tr_path']).iloc[:,[1,3]]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr,hsi_preprocessing= training_config['preprocessing'] , transforms_=hsi_img_transforms)

df_val = pd.read_csv(config['val_path']).iloc[:,[1,3]]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val,hsi_preprocessing= training_config['preprocessing'] , transforms_= val_hsi_transforms)

df_tst = pd.read_csv(config['val_path']).iloc[:,[1,3]]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, hsi_preprocessing= training_config['preprocessing'] , transforms_= val_hsi_transforms)
