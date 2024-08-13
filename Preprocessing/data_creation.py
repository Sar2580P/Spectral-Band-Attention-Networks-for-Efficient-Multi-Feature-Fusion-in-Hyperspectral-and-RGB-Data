import os, sys
import pandas as pd
import json
sys.path.insert(0 , os.getcwd())
from utils import *
import random
random.seed(42)
from sklearn.model_selection import StratifiedKFold

def segment_images():
  class_to_id = {}
  id_to_class = {}
  ct = 0
  rootdir = "dataset_v1"
  final_dir_hsi = "Data/hsi"
  final_dir_rgb = "Data/rgb"
  df_rgb = pd.DataFrame(columns=['path', 'class_id'])
  df_hsi = pd.DataFrame(columns=['path', 'class_id'])

  skip_ct = 0
  for dirpath, dirnames, _ in os.walk(rootdir):

      for dirname in dirnames:
          print(ct , dirname)
          path = os.path.join(dirpath, dirname)
          class_ = dirname
          if class_ not in class_to_id:
            class_to_id[class_] = ct
            id_to_class[ct] = class_
            ct += 1

          for dirpath_, _, filenames in os.walk(path):
            for file in filenames:
              img_path = os.path.join(dirpath_, file)

              if  file.lower().endswith(".jpg"):
                start=0
                if file[-5]=='2':
                  start = 72
                contour_images, count = create_cropping_jpg(img_path)

                for i, img in enumerate(contour_images):
                  cv2.imwrite(os.path.join(final_dir_rgb, '{ct}_{i}.png'.format(ct = ct-1 ,i=i+start)), img)
                  df_rgb.loc[len(df_rgb)] = ['{ct}-{i}.png'.format(ct = ct-1 ,i=i+start), ct-1]

              elif  file.lower().endswith(".bil"):
                start=0
                if file[-5]=='2':
                  start = 72
                img = read_hdr(img_path)
                img = np.array(img.load())
                images = split_image(img, 25, 75, 700, 280, 12, 6)


                for i, seed_image in enumerate(images):
                  name = '{}_{}.npy'.format(ct-1, start+i)
                  try:
                    np.save(os.path.join(final_dir_hsi, name), seed_image)
                    df_hsi.loc[len(df_hsi)] = [name, ct-1]
                  except:
                    print("error in writing hsi images", file)
                    skip_ct += 1


  df_rgb.to_csv('Data/rgb.csv', index = False)
  df_hsi.to_csv('Data/hsi.csv', index = False)
  print("\n\nskipped ", skip_ct, " images")
  mappings = {'class_to_id': class_to_id, 'id_to_class': id_to_class}

  j = json.dumps(mappings, indent=4)
  with open('Data/mappings.json', 'w') as f:
      print(j, file=f)

# segment_images()
#__________________________________________________________________________________________________________________


def create_folds(final_df, variety_count, num_folds=5):
    BASE_PATH = 'Data/' + str(variety_count)

    # Load the variety mappings for the specific variety count
    mapping_json = json.load(open(os.path.join(BASE_PATH, f'{variety_count}_var_mappings.json')))

    # Load the 98 variety mappings
    mapping_json_98 = json.load(open('Data/98/98_var_mappings.json'))

    # Create a mapping from the variety_count class ids to the 98 variety class ids
    class_to_id_98 = mapping_json_98['class_to_id']  # Use 'class_to_id' if that's the correct key
    class_mapping = {int(class_id): int(class_to_id_98[class_name]) for class_id , class_name in mapping_json['id_to_class'].items()}
    assert len(class_mapping) == variety_count , "Mapping not created properly"
    # Filter final_df to include only relevant class IDs
    valid_class_ids = set(class_mapping.values())
    df = final_df[final_df['class_id'].isin(valid_class_ids)].reset_index(drop=True)
    df.rename(columns={'class_id': 'original_class_label'}, inplace=True)

    # Replace class_ids in df with the corresponding keys from class_mapping
    df['class_label'] = df['original_class_label'].map({v: k for k, v in class_mapping.items()})

    skf = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True)
    for fold, (train_index, val_index) in enumerate(skf.split(df, df.loc[:, 'class_label'])):

      if not os.path.exists(os.path.join(BASE_PATH, f'fold_{fold}')):
        os.mkdir(os.path.join(BASE_PATH, f'fold_{fold}'))

      df_train_fold, df_val_fold = df.iloc[train_index, :], df.iloc[val_index, :]

      df_train_fold.to_csv(os.path.join(BASE_PATH, f'fold_{fold}' , 'df_tr.csv') , index = False)
      df_val_fold.to_csv(os.path.join(BASE_PATH, f'fold_{fold}' ,  'df_val.csv') , index = False)
      print('fold_{x} created'.format(x = fold))

from sklearn.model_selection import train_test_split

def data_4_varietal(mapping_4variety, base_mapping, final_df_path='Data/final_df.csv',
                    save_dir='Data/varietal_4', prefix='4_varietal'):
    # Load JSON files
    with open(mapping_4variety) as f:
        class_to_id = json.load(f)['class_to_id']

    with open(base_mapping) as f:
        var_mapping = json.load(f)['class_to_id']

    # Load final_df
    final_df = pd.read_csv(final_df_path)

    # Initialize an empty DataFrame to store the expanded rows
    expanded_df = pd.DataFrame(columns=['class_label', 'hsi_path'])

    # Map img_path based on class_label and expand the DataFrame
    for key, class_id in class_to_id.items():
        # Get the corresponding class_label from var_mapping
        mapped_label = var_mapping.get(key, None)
        if mapped_label is not None:
            # Filter final_df to find image paths
            hsi_paths = final_df[final_df['class_id'] == mapped_label]['hsi_path'].values
            # Create a temporary DataFrame for the current class_id
            temp_df = pd.DataFrame({
                'class_label': [class_id] * len(hsi_paths),
                'hsi_path': hsi_paths
            })
            # Append to the expanded DataFrame
            expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)

    # Split the DataFrame into train, validation, and test sets (70%, 15%, 15%)
    train_df, temp_df = train_test_split(expanded_df, test_size=0.35, random_state=42 , stratify=expanded_df['class_label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42 , stratify=temp_df['class_label'])

    # Save the splits to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, f'{prefix}_train.csv'), index=False)
    val_df.to_csv(os.path.join(save_dir, f'{prefix}_val.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, f'{prefix}_test.csv'), index=False)

    print(f"Data saved to {save_dir} with prefix '{prefix}'")


if not os.path.exists('Data/final_df.csv'):
  hsi_df = pd.read_csv('Data/hsi.csv')

  RGB_BASE_PATH, HSI_BASE_PATH = 'Data/rgb' , 'Data/hsi'
  final_df = pd.DataFrame(columns=['rgb_path', 'hsi_path' , 'class_id'])
  for i in range(len(hsi_df)):

    rgb_path = RGB_BASE_PATH +'/'+ hsi_df.iloc[i,0][:-4]+'.png'
    hsi_path = HSI_BASE_PATH +'/'+ hsi_df.iloc[i,0]
    class_id = hsi_df.iloc[i,1]
    final_df.loc[len(final_df)] = [rgb_path , hsi_path , class_id]

  final_df.to_csv('Data/final_df.csv' , index = False)


if __name__ == '__main__':
  li = [12,24,37 , 55 , 75 , 96 , 98]
  for class_count in li:
    create_folds(pd.read_csv('Data/final_df.csv'), class_count)


  # data_4_varietal(mapping_4variety='Data/varietal_4/4_varietal.json' , base_mapping='Data/98/98_var_mappings.json')

