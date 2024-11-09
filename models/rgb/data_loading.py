import pandas as pd
from models.train_eval import MyDataset
from processing.utils import read_yaml
from models.transforms import rgb_transforms, val_rgb_transforms

config = read_yaml('models/rgb/config.yaml')
data_config = {
  'tr_path' : 'Data/{class_ct}/fold_{fold}/df_tr.csv'.format(class_ct=config['num_classes'], fold = config['fold']),
  'val_path' : 'Data/{class_ct}/fold_{fold}/df_val.csv'.format(class_ct=config['num_classes'], fold = config['fold']),
  # 'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
}



df_tr = pd.read_csv(data_config['tr_path']).loc[:,['rgb_path' , 'class_label']]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr, transforms_=rgb_transforms , data_dir=config['data_dir'])

df_val = pd.read_csv(data_config['val_path']).loc[:,['rgb_path' , 'class_label']]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val, transforms_= val_rgb_transforms , data_dir=config['data_dir'])

df_tst = pd.read_csv(data_config['val_path']).loc[:,['rgb_path' , 'class_label']]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, transforms_=val_rgb_transforms ,  data_dir=config['data_dir'])
