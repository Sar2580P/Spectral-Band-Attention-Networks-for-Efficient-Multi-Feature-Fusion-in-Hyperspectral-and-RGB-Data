import os
import torch
import torch.nn as nn
from processing.utils import read_yaml
from models.train_eval import Classifier
from torch.utils.data import DataLoader
from models.model_architectures import RGB_Resnet , DenseNetRGB , GoogleNet, DenseNet
from typing import List, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.train_eval import MyDataset
from models.transforms import val_hsi_transforms, val_rgb_transforms
import glob
hsi_config = read_yaml('models/hsi/config.yaml')
rgb_config = read_yaml('models/rgb/config.yaml')

MODEL_MAP ={
    "rgb" :{
        'resnet' : RGB_Resnet,
        'densenet' : DenseNetRGB,
        'google_net' : GoogleNet,
        } ,
    "hsi" : {
        'densenet' : DenseNet
    }
}


def load_model(modality:str , mode_name:str) -> nn.Module:
    assert modality in ['rgb', 'hsi'] , "modality should be either 'rgb' or 'hsi'"
    config = rgb_config if modality == 'rgb' else hsi_config
    config['model_name'] = mode_name
    model_obj =  MODEL_MAP[modality][config['model_name']](config = config)

    # load checkpoint
    ckpt_dir = f"results/{modality}/classes-{config['num_classes']}/fold-{config['fold']}/ckpts"
    if modality=='hsi':
        ckpt_path = glob.glob(os.path.join(ckpt_dir, f"densenet-18-24-16-10__{config['data_type']}__{config['preprocessing']}__{config['C']}--*.ckpt"))[0]
    else :
        ckpt_file = [f for f in os.listdir(ckpt_dir) if f.startswith(config['model_name'])][0]
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    classifier = Classifier.load_from_checkpoint(ckpt_path, model_obj=model_obj)
    return classifier.model_obj.model


def get_layer_output(model : nn.Module ,  batch : torch.Tensor , layer_name:str=None)->torch.Tensor:
    '''
        To get output from a specific layer of the model
    '''

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    layer_output = []
    def hook(module, input, output):
        layer_output.append(output)
    if layer_name:
        layer = model._modules.get(layer_name)
        hook_handle = layer.register_forward_hook(hook)
    else:
        hook_handle = model.register_forward_hook(hook)
    with torch.no_grad():
        x, y = batch
        model(x.to(device))
    hook_handle.remove()

    #attach labels colto the output
    res = torch.cat([torch.cat(layer_output, dim=0) , batch[1].unsqueeze(1).to(device=device)] , dim=1)   # [batch_size, num_features + 1]
    return res


def get_loaders(fold:int, num_classes:int)->Tuple[Tuple[DataLoader,DataLoader], Tuple[DataLoader,DataLoader]]:
    rgb_config['fold'] = hsi_config['fold'] = fold

    # dataset
    data_config = {
                'tr_path' : f"Data/{num_classes}/fold_{fold}/df_tr.csv" ,
                'val_path' : f"Data/{num_classes}/fold_{fold}/df_val.csv"
                # 'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
                }
    df_tr = pd.read_csv(data_config['tr_path'])
    df_val = pd.read_csv(data_config['val_path'])

    hsi_tr_dataset = MyDataset(df_tr.rename(columns={'hsi_path':'img_path' , 'class_label':'class_id'}) ,
                               hsi_preprocessing= hsi_config['preprocessing'] , transforms_=val_hsi_transforms ,
                               data_dir = hsi_config['data_dir'])
    hsi_val_dataset = MyDataset(df_val.rename(columns={'hsi_path':'img_path' , 'class_label':'class_id'}) ,
                                hsi_preprocessing= hsi_config['preprocessing'] , transforms_= val_hsi_transforms,
                                data_dir = hsi_config['data_dir'])

    rgb_tr_dataset = MyDataset(df_tr.rename(columns={'rgb_path':'img_path' , 'class_label':'class_id'}),
                               transforms_=val_rgb_transforms , data_dir=rgb_config['data_dir'])
    rgb_val_dataset = MyDataset(df_val.rename(columns={'rgb_path':'img_path' , 'class_label':'class_id'}),
                                transforms_= val_rgb_transforms , data_dir=rgb_config['data_dir'])

    # data-loaders
    hsi_tr_loader = DataLoader(hsi_tr_dataset, batch_size=hsi_config['BATCH_SIZE'], shuffle=False, num_workers=hsi_config['num_workers'])
    hsi_val_loader = DataLoader(hsi_val_dataset, batch_size=hsi_config['BATCH_SIZE'], shuffle=False, num_workers=hsi_config['num_workers'])

    rgb_tr_loader = DataLoader(rgb_tr_dataset, batch_size=rgb_config['BATCH_SIZE'], shuffle=False, num_workers=rgb_config['num_workers'])
    rgb_val_loader = DataLoader(rgb_val_dataset, batch_size=rgb_config['BATCH_SIZE'], shuffle=False, num_workers=rgb_config['num_workers'])

    return (hsi_tr_loader, hsi_val_loader), (rgb_tr_loader, rgb_val_loader)


def get_base_models_predictions(rgb_models: List[str], hsi_models : List[str], num_classes : int) :
    models =[]
    for model_name in rgb_models:
        models.append(load_model('rgb', model_name))
    for model_name in hsi_models:
        models.append(load_model('hsi', model_name))

    SAVE_DIR = f'results/ensemble/base_models/classes-{num_classes}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    rgb_config['num_classes'] = hsi_config['num_classes'] = num_classes

    for fold in range(5):
        os.makedirs(os.path.join(SAVE_DIR, f'fold_{fold}'), exist_ok=True)

        (hsi_tr_loader, hsi_val_loader), (rgb_tr_loader, rgb_val_loader) = get_loaders(fold, num_classes)
        # iterate through all models and over all data
        for idx , (model_name , model) in enumerate(zip(rgb_models+hsi_models , models)):

            modality = 'rgb' if idx < len(rgb_models) else 'hsi'

            loaders = [hsi_tr_loader , hsi_val_loader] if modality == 'hsi' else [rgb_tr_loader , rgb_val_loader]
            for i , dataloader in enumerate(loaders):
                predictions = None
                for batch in tqdm(dataloader , desc = f"Batches for {model_name}, modality: {modality}"):
                    predictions = get_layer_output(model, batch) if predictions is None  \
                                    else torch.cat([predictions, get_layer_output(model, batch)], dim=0)

                predictions = predictions.detach().cpu().numpy()
                if modality=='rgb' :
                    prediction_save_path = os.path.join(SAVE_DIR, f'fold_{fold}',
                                                        f"{modality}_{model_name}_{'train' if i==0 else 'val'}.npy")
                else :
                    prediction_save_path = os.path.join(SAVE_DIR, f'fold_{fold}',
                                                        f"{modality}_{model_name}_{'train' if i==0 else 'val'}_{hsi_config['data_type']}__{hsi_config['preprocessing']}__{hsi_config['C']}.npy")

                np.save(prediction_save_path, predictions)
            print(f"Saved {model_name}-{modality} predictions for fold {fold}")


        print(f"Saved base model predictions for fold {fold}")
    return


if __name__=="__main__":
    get_base_models_predictions([] , ['densenet'], 96)
    print("Done")

    # path = 'results/ensemble/base_models/classes-96/fold_0/rgb_densenet_val.npy'
    # data = np.load(path)
    # print(data[0])