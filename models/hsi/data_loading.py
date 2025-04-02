import pandas as pd
from models.train_eval import MyDataset
from models.transforms import hsi_img_transforms, val_hsi_transforms
from torch.utils.data import DataLoader

def get_hsi_loaders(config):
  data_config = {
    'tr_path' : f"Data/{config['num_classes']}/df_train.csv",
    'val_path' : f"Data/{config['num_classes']}/df_val.csv",
    'tst_path' : f"Data/{config['num_classes']}/df_test1.csv"
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
  
  tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['num_workers'])
  val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])
  tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['num_workers'])
  
  return tr_loader, val_loader, tst_loader



