import pandas as pd
from models.train_eval import MyDataset
from processing.utils import read_yaml
from models.transforms import hsi_img_transforms, val_hsi_transforms
config=read_yaml('models/hsi/config.yaml')

data_config = {
  'tr_path' : 'Data/{class_ct}/df_train.csv'.format(class_ct=config['num_classes']),
  'val_path' : 'Data/{class_ct}/df_val.csv'.format(class_ct=config['num_classes']),
  'tst_path' : 'Data/{class_ct}/df_test1.csv'.format(class_ct=config['num_classes']),
}

df_tr = pd.read_csv(data_config['tr_path']).loc[:,['hsi_path' , 'class_label']]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr,hsi_preprocessing= config['preprocessing'] , transforms_=hsi_img_transforms ,
                       data_dir = config['data_dir'])

df_val = pd.read_csv(data_config['val_path']).loc[:,['hsi_path' , 'class_label']]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val,hsi_preprocessing= config['preprocessing'] , transforms_= val_hsi_transforms,
                        data_dir = config['data_dir'])

df_tst = pd.read_csv(data_config['tst_path']).loc[:,['hsi_path' , 'class_label']]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, hsi_preprocessing= config['preprocessing'] , transforms_= val_hsi_transforms ,
                        data_dir = config['data_dir'])

