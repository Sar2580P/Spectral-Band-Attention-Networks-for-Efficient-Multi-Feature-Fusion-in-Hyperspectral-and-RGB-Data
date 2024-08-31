import pandas as pd
from models.train_eval import MyDataset
from processing.utils import read_yaml
from torchvision import transforms

config=read_yaml('models/hsi/config.yaml')

data_config = {
  'tr_path' : 'Data/{class_ct}/fold_{fold}/df_tr.csv'.format(class_ct=config['num_classes'], fold = config['fold']),
  'val_path' : 'Data/{class_ct}/fold_{fold}/df_val.csv'.format(class_ct=config['num_classes'], fold = config['fold']),
  # 'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
}

hsi_img_transforms = transforms.Compose([
   transforms.ToTensor(),
   transforms.RandomHorizontalFlip(p=0.7),
   transforms.RandomVerticalFlip(p=0.7),
   transforms.RandomAffine(degrees=5, translate=(0.1, 0.2), scale=(0.8, 1.25)),

])
val_hsi_transforms = transforms.Compose([
   transforms.ToTensor(),
])


df_tr = pd.read_csv(data_config['tr_path']).iloc[:,[1,3]]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr,hsi_preprocessing= config['preprocessing'] , transforms_=hsi_img_transforms , data_dir = config['data_dir'])

df_val = pd.read_csv(data_config['val_path']).iloc[:,[1,3]]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val,hsi_preprocessing= config['preprocessing'] , transforms_= val_hsi_transforms,  data_dir = config['data_dir'])

df_tst = pd.read_csv(data_config['val_path']).iloc[:,[1,3]]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, hsi_preprocessing= config['preprocessing'] , transforms_= val_hsi_transforms , data_dir = config['data_dir'])
