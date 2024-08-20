import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# resnet_conf_mat = 'models/rgb/resnet/confusion_matrix.csv'
# enet_conf_mat = 'models/rgb/enet/confusion_matrix.csv'
gnet_conf_mat = 'models/rgb/google_net/confusion_matrix.csv'
# xception_conf_mat = 'models/hsi/xception/confusion_matrix.csv'
denseNet_conf_mat = 'models/hsi/dense_net/confusion_matrix.csv' 

# df_resnet = pd.read_csv(resnet_conf_mat)
# df_enet = pd.read_csv(enet_conf_mat)
df_gnet = pd.read_csv(gnet_conf_mat)
# df_xception = pd.read_csv(xception_conf_mat)
df_denseNet = pd.read_csv(denseNet_conf_mat)

def calculate_in_class_accuracies(df):
    df.drop('Unnamed: 0' , axis=1, inplace=True)
    confusion_matrix = df.to_numpy()

    # Calculate the diagonal of the confusion matrix.
    diagonal = np.diag(confusion_matrix)
    # Calculate the total number of predictions for each class.
    class_totals = np.sum(confusion_matrix, axis=1)

    # Calculate the in-class accuracies.
    in_class_accuracies = diagonal / class_totals

    return in_class_accuracies

# in_class_accuracies_resnet = calculate_in_class_accuracies(df_resnet)
# in_class_accuracies_enet = calculate_in_class_accuracies(df_enet)
in_class_accuracies_gnet = calculate_in_class_accuracies(df_gnet)
# in_class_accuracies_xception = calculate_in_class_accuracies(df_xception)
in_class_accuracies_denseNet = calculate_in_class_accuracies(df_denseNet)


# print(in_class_accuracies_resnet.shape)
# print(in_class_accuracies_enet.shape)
print(in_class_accuracies_gnet.shape)
# print(in_class_accuracies_xception.shape)
print(in_class_accuracies_denseNet.shape)

arr = np.stack((in_class_accuracies_denseNet,  
                 in_class_accuracies_gnet)).transpose()
# 
fig, axs = plt.subplots(96, 1, figsize=(20, 200))
fig.tight_layout(pad=7.0)
models = ['DenseNet','GoogleNet']  # 

for i , per_class_acc in enumerate(arr):
    axs[i].barh(models,per_class_acc, color = ['red', 'green'])
    axs[i].yaxis.set_tick_params(pad=10)
    axs[i].set_xlim([0,1])
    axs[i].set_xlabel('Accuracy')
    axs[i].set_title(f'Class {i} accuracy')
    axs[i].yaxis.set_tick_params(pad=10)

plt.savefig('per_class_accuracies.png')
