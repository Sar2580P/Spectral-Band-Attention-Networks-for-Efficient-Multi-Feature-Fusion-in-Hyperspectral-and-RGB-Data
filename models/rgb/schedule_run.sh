#!/bin/bash

# Define the range of folds and the list of num_classes
folds=(4)
num_classes=(12 24 37 55 75 96 98)  # New requirement only for num_classes=96

# Define the list of runner files and their corresponding config files
runner_files=("models/rgb/densenet/denseNetRGB.py" "models/rgb/google_net/googleNet.py" "models/rgb/resnet-34/resnet34_RGB.py" "models/rgb/resnet-50/resnet50_RGB.py")
config_files=("models/rgb/densenet/config.yaml" "models/rgb/google_net/config.yaml" "models/rgb/resnet-34/config.yaml" "models/rgb/resnet-50/config.yaml")

# Path to the dataloading.py file
dataloading_file="models/rgb/data_loading.py"

# Function to update the YAML file
update_yaml() {
    python -c "
import yaml

# Load YAML file
with open('$1', 'r') as file:
    config = yaml.safe_load(file)

# Update values
config['fold'] = $2
config['num_classes'] = $3

# Save updated YAML file
with open('$1', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3
}

# Function to update the dataloading.py file
update_dataloading() {
    python -c "
file_path = '$dataloading_file'

# Read the file
with open(file_path, 'r') as file:
    data = file.readlines()

# Update the fold and num_classes lines
for i, line in enumerate(data):
    if line.startswith('fold ='):
        data[i] = f'fold = $1\n'
    elif line.startswith('num_classes ='):
        data[i] = f'num_classes = $2\n'

# Write the updated file back
with open(file_path, 'w') as file:
    file.writelines(data)
    " $1 $2
}

# Iterate over each combination of fold and num_classes
for fold in "${folds[@]}"; do
    for num_class in "${num_classes[@]}"; do
        for index in "${!runner_files[@]}"; do
            runner_file=${runner_files[$index]}
            config_file=${config_files[$index]}

            echo "Updating $config_file and dataloading.py and running $runner_file with fold=$fold and num_classes=$num_class"

            # Update the YAML config file with current fold and num_class
            update_yaml $config_file $fold $num_class

            # Update the dataloading.py file with current fold and num_class
            update_dataloading $fold $num_class

            # Run the Python script
            python $runner_file &

            # Wait for the previous run to complete
            wait

            echo "Completed $runner_file with fold=$fold and num_classes=$num_class"
        done
    done
done

echo "All combinations have been processed."
