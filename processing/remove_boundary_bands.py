import os
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

Original_HSI_DIR = 'Data/hsi'
New_HSI_DIR = 'Data/hsi_trimmed'

# removing starting 10 bands and ending 6 bands of each hsi image


if not os.path.exists(New_HSI_DIR):
    os.makedirs(New_HSI_DIR)
    for root_dir, _, files in (os.walk(Original_HSI_DIR)):
        for file in tqdm.tqdm(files):
            mat = np.load(os.path.join(root_dir, file))
            mat = mat[:, :, 14:-7]
            np.save(os.path.join(New_HSI_DIR, file), mat)

# df = pd.read_csv('Data/98/fold_0/df_tr.csv').sample(n = 10, replace = False )

# fig , axs = plt.subplots(4, 10, figsize = (20, 10))

# for i, img_path in enumerate(df['hsi_path']):
#     img = np.load(img_path)
#     axs[0, i].imshow(img[:,:,0])
#     axs[0,i].set_title('New HSI : 0')

#     axs[1, i].imshow(img[:,:,-1])
#     axs[1, i].set_title('New HSI : -1')

#     original_img_path = os.path.join(Original_HSI_DIR, img_path.split('/')[-1])
#     original_img = np.load(original_img_path)
#     axs[2, i].imshow(original_img[:,:,0])
#     axs[2,i].set_title('Original HSI : 0')

#     axs[3, i].imshow(original_img[:,:,-1])
#     axs[3, i].set_title('Original HSI : -1')

# plt.savefig('hsi_channel_analysis.png')
