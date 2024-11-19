#!/bin/bash

# Define the range of folds, list of num_classes, etc.
folds=(0 1 2 3 4)
num_classes=(96)  # (12 24 37 55 75 96)
preprocessing=('none')
data_dir=('Data/hsi_masked_spa/Channel_25' 'Data/hsi_masked_spa/Channel_50' 'Data/hsi_masked_spa/Channel_75' 'Data/hsi_masked_spa/Channel_100' 'Data/hsi_masked_spa/Channel_125' 'Data/hsi_masked_spa/Channel_150' 'Data/hsi_masked_spa/Channel_168')
C=(25 50 75 100 125 150 168)
data_type=('maskedBAM')
model_name=('resnet' 'densenet' 'google_net')

hsi_yaml="models/hsi/config.yaml"
rgb_yaml="models/rgb/config.yaml"
# runner_file="models/ensemble/base.py"  # Path to your Python script
# runner_file="models/ensemble/deep_ensemble.py"
runner_file="models/ensemble/regression_ensemble.py"

# Function to update the HSI YAML file
update_hsi_yaml() {
    python -c "
import yaml
with open('$hsi_yaml', 'r') as file:
    config = yaml.safe_load(file)
config['fold'] = $1
config['num_classes'] = $2
config['preprocessing'] = '$3'
config['data_dir'] = '$4'
config['C'] = $5
config['data_type'] = '$6'
with open('$hsi_yaml', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3 $4 $5 $6
}

# Function to update the RGB YAML file
update_rgb_yaml() {
    python -c "
import yaml
with open('$rgb_yaml', 'r') as file:
    config = yaml.safe_load(file)
config['fold'] = $1
config['num_classes'] = $2
config['model_name'] = '$3'
with open('$rgb_yaml', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3
}

# Iterate over each combination of parameters for HSI configuration
for fold in "${folds[@]}"; do
    for num_class in "${num_classes[@]}"; do
        for preprocessing in "${preprocessing[@]}"; do
            for i in "${!data_dir[@]}"; do
                echo "Updating HSI YAML and running $runner_file with fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[0]}"

                # Update the HSI YAML file with current parameters
                update_hsi_yaml $fold $num_class "$preprocessing" "${data_dir[$i]}" "${C[$i]}" "${data_type[0]}"

                # Run the Python script and check for errors
                python $runner_file
                if [ $? -ne 0 ]; then
                    echo "Error occurred while running the HSI trainer with fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[0]}"
                    exit 1
                fi
                echo "Completed HSI fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[0]}"
            done
        done
    done
done

# # Iterate over each combination of parameters for RGB configuration
# for fold in "${folds[@]}"; do
#     for num_class in "${num_classes[@]}"; do
#         for model in "${model_name[@]}"; do
#             echo "Updating RGB YAML and running $runner_file with fold=$fold, num_classes=$num_class, model_name=$model"

#             # Update the RGB YAML file with current parameters
#             update_rgb_yaml $fold $num_class "$model"

#             # Run the Python script and check for errors
#             python $runner_file
#             if [ $? -ne 0 ]; then
#                 echo "Error occurred while running the RGB trainer with fold=$fold, num_classes=$num_class, model_name=$model"
#                 exit 1
#             fi
#             echo "Completed RGB fold=$fold, num_classes=$num_class, model_name=$model"
#         done
#     done
# done

echo "All combinations have been processed."
