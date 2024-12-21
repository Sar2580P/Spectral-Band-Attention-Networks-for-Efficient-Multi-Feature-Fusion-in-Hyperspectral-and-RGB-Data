import os
import pandas as pd
import json
import random
random.seed(42)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_splits(final_df, variety_count):
    BASE_PATH = 'Data/' + str(variety_count)

    # Load the variety mappings for the specific variety count
    mapping_json = json.load(open(os.path.join(BASE_PATH, f'{variety_count}_var_mappings.json')))

    # Load the 98 variety mappings
    mapping_json_98 = json.load(open('Data/98/98_var_mappings.json'))

    # Create a mapping from the variety_count class ids to the 98 variety class ids
    class_to_id_98 = mapping_json_98['class_to_id']  # Use 'class_to_id' if that's the correct key
    class_mapping = {int(class_id): int(class_to_id_98[class_name]) for class_id, class_name in mapping_json['id_to_class'].items()}
    assert len(class_mapping) == variety_count, "Mapping not created properly"

    # Filter final_df to include only relevant class IDs
    valid_class_ids = set(class_mapping.values())
    df = final_df[final_df['class_id'].isin(valid_class_ids)].reset_index(drop=True)
    df.rename(columns={'class_id': 'original_class_label'}, inplace=True)

    # Replace class_ids in df with the corresponding keys from class_mapping
    df['class_label'] = df['original_class_label'].map({v: k for k, v in class_mapping.items()})
    df['stratify_col(class_label+plate_count)'] = df['class_label'].astype(str) + '_' + df['plate_count'].astype(str)

    # Split the data into train, val, test1, test2
    train_df, test_combined_df = train_test_split(df, test_size=0.4, stratify=df['stratify_col(class_label+plate_count)'], random_state=7)

    # Split the test_combined_df into val, test1, test2
    val_df, test1_df = train_test_split(test_combined_df, test_size=0.7, stratify=test_combined_df['stratify_col(class_label+plate_count)'], random_state=7)
    test2_df, test1_df = train_test_split(test1_df, test_size=0.5, stratify=test1_df['stratify_col(class_label+plate_count)'], random_state=7)

    # Create directories for each split
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    # Save the splits
    for split, split_df in zip(['train', 'val', 'test1', 'test2'], [train_df, val_df, test1_df, test2_df]):
        split_df.to_csv(os.path.join(BASE_PATH, f'df_{split}.csv'), index=False)
        print(f'{split}_split created')

if __name__ == '__main__':
    if not os.path.exists('Data/final_df.csv'):
        hsi_df = pd.read_csv('Data/hsi.csv')
        final_df = pd.DataFrame(columns=['rgb_path', 'hsi_path', 'class_id', 'plate_count'])
        for i in tqdm(range(len(hsi_df)), desc='Creating final df with all files'):
            rgb_path = hsi_df.iloc[i, 0][:-4] + '.png'
            plate_count = int(hsi_df.iloc[i, 0][:-4].split('_')[1]) // 72  # plate_count=0 for 1st plate, 1 for 2nd plate
            hsi_path = hsi_df.iloc[i, 0]
            class_id = hsi_df.iloc[i, 1]
            final_df.loc[len(final_df)] = [rgb_path, hsi_path, class_id, plate_count]

        final_df.to_csv('Data/final_df.csv', index=False)

    li = [96]
    for class_count in li:
        create_splits(pd.read_csv('Data/final_df.csv'), class_count)