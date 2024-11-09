#!/bin/bash

# Define the range of folds and the list of num_classes
folds=(0 1 2 3 4)
num_classes=(96)  #(12 24 37 55 75 96)
preprocessing=('none')
data_dir=('Data/hsi_masked_pcaLoading/Channel_25' 'Data/hsi_masked_pcaLoading/Channel_50' 'Data/hsi_masked_pcaLoading/Channel_75' 'Data/hsi_masked_pcaLoading/Channel_100' 'Data/hsi_masked_pcaLoading/Channel_125' 'Data/hsi_masked_pcaLoading/Channel_150' 'Data/hsi_masked_pcaLoading/Channel_168')
C=(25 50 75 100 125 150 168)
data_type=('maskedPCALoading')  # , 'maskedBAM' 'pcaLoading', 'maskedSPA'

yaml_file="models/hsi/config.yaml"  # Path to your YAML file

# Function to update the YAML file
update_yaml() {
    python -c "
import yaml

# Load YAML file
with open('$yaml_file', 'r') as file:
    config = yaml.safe_load(file)

# Update values
config['fold'] = $1
config['num_classes'] = $2
config['preprocessing'] = '$3'
config['data_dir'] = '$4'
config['C'] = $5
config['data_type'] = '$6'

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3 $4 $5 $6
}

runner_file="models/hsi/trainer.py"  # Path to your Python script

# Iterate over each combination of fold and num_classes
for fold in "${folds[@]}"; do
    for num_class in "${num_classes[@]}"; do
        for preprocessing in "${preprocessing[@]}"; do
            for i in "${!data_dir[@]}"; do
                echo "Updating YAML and running $runner_file with fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[0]}"

                # Update the YAML file with current parameters
                update_yaml $fold $num_class "$preprocessing" "${data_dir[$i]}" "${C[$i]}" "${data_type[0]}"

                # Run the Python script and wait for it to complete
                python $runner_file
                if [ $? -ne 0 ]; then
                    echo "Error occurred while running the trainer with fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[0]}"
                    exit 1
                fi

                echo "Completed fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[0]}"
            done
        done
    done
done

echo "All combinations have been processed."
