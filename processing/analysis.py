import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

df = pd.read_csv('Data/98/fold_0/df_tr.csv')

ct = 15

# df1 = df.sample(n=ct)
# df1.to_csv('test.csv')
df1 = pd.read_csv('test.csv')
from torchvision import transforms
hsi_img_transforms = transforms.Compose([
   transforms.ToTensor(), 
   transforms.RandomAffine(degrees = 0 , translate=(0.1, 0.1),  scale=(0.8,1.2)),
   
])
def apply_transform(img, ax):
    img = hsi_img_transforms(img)
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    # print(img.shape)
    ax.imshow(img[:,:,50])
    
fig, axs = plt.subplots(1,ct, figsize=(20,20))    
    
for i ,img_path in enumerate(df1['hsi_path']):
    img = np.load(img_path)
    apply_transform(img , axs[i])
    
plt.savefig('test2.png')    