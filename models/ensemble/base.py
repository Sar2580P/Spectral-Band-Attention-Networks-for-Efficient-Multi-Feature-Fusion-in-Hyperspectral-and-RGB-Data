import os
import torch
import torch.nn as nn
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

from omegaconf import OmegaConf
import os


def read_yaml(file_path):
    conf = OmegaConf.load(file_path)
    config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    return config

hsi_config = read_yaml('models/hsi/config.yaml')
rgb_config = read_yaml('models/rgb/config.yaml')

MODEL_MAP ={
    "rgb" :{
        'resnet' : RGB_Resnet,
        'densenet' : DenseNetRGB,
        'google_net' : GoogleNet,
        } ,
    "hsi" : {
        'densenet' : DenseNet,
        'densenet__withoutTemp_LossMean' : DenseNet
    }
}


def load_model(modality:str , mode_name:str) -> nn.Module:
    assert modality in ['rgb', 'hsi'] , "modality should be either 'rgb' or 'hsi'"
    config = rgb_config if modality == 'rgb' else hsi_config
    config['model_name'] = mode_name
    model_obj =  MODEL_MAP[modality][config['model_name']](config = config)

    # load checkpoint
    ckpt_dir = f"results/{modality}/classes-{config['num_classes']}/ckpts"
    if modality=='hsi':
        ckpt_path = glob.glob(os.path.join(ckpt_dir, f"{mode_name}-24-18-16-10*__{config['data_type']}__{config['preprocessing']}__{config['C']}--*.ckpt"))[0]
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


def get_loaders(num_classes:int)->Tuple[Tuple[DataLoader,DataLoader], Tuple[DataLoader,DataLoader]]:
    # dataset
    data_config = {
    'tr_path' : 'Data/{class_ct}/df_train.csv'.format(class_ct=num_classes),
    'val_path' : 'Data/{class_ct}/df_val.csv'.format(class_ct=num_classes),
    'tst_path' : 'Data/{class_ct}/df_test1.csv'.format(class_ct=num_classes),
    }
    df_tr = pd.read_csv(data_config['tr_path'])
    df_val = pd.read_csv(data_config['val_path'])
    df_tst = pd.read_csv(data_config['tst_path'])

    hsi_tr_dataset = MyDataset(df_tr.rename(columns={'hsi_path':'img_path' , 'class_label':'class_id'}) ,
                               hsi_preprocessing= hsi_config['preprocessing'] , transforms_=val_hsi_transforms ,
                               data_dir = hsi_config['data_dir'])
    hsi_val_dataset = MyDataset(df_val.rename(columns={'hsi_path':'img_path' , 'class_label':'class_id'}) ,
                                hsi_preprocessing= hsi_config['preprocessing'] , transforms_= val_hsi_transforms,
                                data_dir = hsi_config['data_dir'])
    hsi_tst_dataset = MyDataset(df_tst.rename(columns={'hsi_path':'img_path' , 'class_label':'class_id'}) ,
                                hsi_preprocessing= hsi_config['preprocessing'] , transforms_= val_hsi_transforms,
                                data_dir = hsi_config['data_dir'])

    rgb_tr_dataset = MyDataset(df_tr.rename(columns={'rgb_path':'img_path' , 'class_label':'class_id'}),
                               transforms_=val_rgb_transforms , data_dir=rgb_config['data_dir'])
    rgb_val_dataset = MyDataset(df_val.rename(columns={'rgb_path':'img_path' , 'class_label':'class_id'}),
                                transforms_= val_rgb_transforms , data_dir=rgb_config['data_dir'])
    rgb_tst_dataset = MyDataset(pd.read_csv(data_config['tst_path']).rename(columns={'rgb_path':'img_path' , 'class_label':'class_id'}),
                                transforms_= val_rgb_transforms , data_dir=rgb_config['data_dir'])

    # data-loaders
    hsi_tr_loader = DataLoader(hsi_tr_dataset, batch_size=hsi_config['BATCH_SIZE'], shuffle=False, num_workers=hsi_config['num_workers'])
    hsi_val_loader = DataLoader(hsi_val_dataset, batch_size=hsi_config['BATCH_SIZE'], shuffle=False, num_workers=hsi_config['num_workers'])
    hsi_tst_loader = DataLoader(hsi_tst_dataset, batch_size=hsi_config['BATCH_SIZE'], shuffle=False, num_workers=hsi_config['num_workers'])

    rgb_tr_loader = DataLoader(rgb_tr_dataset, batch_size=rgb_config['BATCH_SIZE'], shuffle=False, num_workers=rgb_config['num_workers'])
    rgb_val_loader = DataLoader(rgb_val_dataset, batch_size=rgb_config['BATCH_SIZE'], shuffle=False, num_workers=rgb_config['num_workers'])
    rgb_tst_loader = DataLoader(rgb_tst_dataset, batch_size=rgb_config['BATCH_SIZE'], shuffle=False, num_workers=rgb_config['num_workers'])

    return (hsi_tr_loader, hsi_val_loader, hsi_tst_loader), (rgb_tr_loader, rgb_val_loader, rgb_tst_loader)

def compute_accuracy(output_matrix):
    """
    Computes the accuracy of a classification model given the logits and actual labels.

    Parameters:
    - output_matrix (numpy.ndarray): A 2D matrix of shape (num_samples, num_classes+1),
                                      where the last column contains the actual class labels
                                      and the rest are the logits for the num_classes.

    Returns:
    - float: The accuracy of the model.
    """
    # Extract logits and actual labels
    logits = output_matrix[:, :-1]
    actual_labels = output_matrix[:, -1].astype(int)

    # Predicted class is the one with the highest logit value
    predicted_labels = np.argmax(logits, axis=1)

    # Calculate accuracy
    correct_predictions = np.sum(predicted_labels == actual_labels)
    total_samples = output_matrix.shape[0]
    accuracy = correct_predictions / total_samples

    return accuracy


def get_base_models_predictions(rgb_models: List[str], hsi_models : List[str], num_classes : int) :
    models =[]
    for model_name in rgb_models:
        try:
            models.append(load_model('rgb', model_name))
        except Exception as e:
            print("ERROR", e)
    for model_name in hsi_models:
        try :
            models.append(load_model('hsi', model_name))
        except Exception as e:
            print("ERROR", e)


    SAVE_DIR = f'results/ensemble/base_models/classes-{num_classes}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    rgb_config['num_classes'] = hsi_config['num_classes'] = num_classes


    (hsi_tr_loader, hsi_val_loader, hsi_tst_loader), (rgb_tr_loader, rgb_val_loader, rgb_tst_loader) = get_loaders( num_classes)
    # iterate through all models and over all data
    print(rgb_models+hsi_models , len(models))
    for idx , (model_name , model) in enumerate(zip(rgb_models+hsi_models , models)):

        modality = 'rgb' if idx < len(rgb_models) else 'hsi'

        loaders = [hsi_tr_loader , hsi_val_loader, hsi_tst_loader] if modality == 'hsi' else [rgb_tr_loader , rgb_val_loader, rgb_tst_loader]
        for i , (dataloader, loader_name) in enumerate(zip(loaders, ['train', 'val', 'tst'])):
            predictions = None
            for batch in tqdm(dataloader , desc = f"Batches for {model_name}, modality: {modality}"):
                predictions = get_layer_output(model, batch) if predictions is None  \
                                else torch.cat([predictions, get_layer_output(model, batch)], dim=0)

            predictions = predictions.detach().cpu().numpy()
            if modality=='rgb' :
                prediction_save_path = os.path.join(SAVE_DIR,
                                                    f"{modality}_{model_name}_{loader_name}.npy")
            else :
                prediction_save_path = os.path.join(SAVE_DIR,
                                                    f"{modality}_{model_name}_{loader_name}_{hsi_config['data_type']}__{hsi_config['preprocessing']}__{hsi_config['C']}.npy")
            print(f"Accuracy for {model_name} on {loader_name} data: {compute_accuracy(predictions):.2%}")
            np.save(prediction_save_path, predictions)
    return


if __name__=="__main__":
    get_base_models_predictions([] , ['densenet'], 96)  # 'resnet', 'densenet', 'google_net'
    print("Done")

    # path = 'results/ensemble/base_models/classes-96/fold_0/rgb_densenet_val.npy'
    # data = np.load(path)
    # print(data[0])