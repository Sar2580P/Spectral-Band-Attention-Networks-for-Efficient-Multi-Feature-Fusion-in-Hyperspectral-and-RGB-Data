import pandas as pd
from train_eval import MyDataset
from processing.utils import read_yaml
from torchvision import transforms

config = read_yaml('models/rgb/config.yaml')
data_config = {
  'tr_path' : 'Data/{class_ct}/fold_{fold}/df_tr.csv'.format(class_ct=config['num_classes'], fold = config['fold']),
  'val_path' : 'Data/{class_ct}/fold_{fold}/df_val.csv'.format(class_ct=config['num_classes'], fold = config['fold']),
  # 'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
}

rgb_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.25)),
      transforms.ToTensor(),
   ])

val_rgb_transforms = transforms.Compose([
      transforms.ToTensor(),
   ])

df_tr = pd.read_csv(config['tr_path']).iloc[:,[0,3]]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr, transforms_=rgb_transforms , data_dir=config['data_dir'])

df_val = pd.read_csv(config['val_path']).iloc[:,[0,3]]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val, transforms_= val_rgb_transforms , data_dir=config['data_dir'])

df_tst = pd.read_csv(config['val_path']).iloc[:,[0,3]]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, transforms_=val_rgb_transforms ,  data_dir=config['data_dir'])
