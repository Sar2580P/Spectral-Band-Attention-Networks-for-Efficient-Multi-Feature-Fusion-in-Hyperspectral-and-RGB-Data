#!/bin/bash

# Define the list of num_classes
num_classes=(96)
preprocessing=('none')
data_dir=('Data/hsi_masked_bam/channels_50' 'Data/hsi_masked_bam/channels_75' 'Data/hsi_masked_bam/channels_100' 'Data/hsi_masked_bam/channels_125' 'Data/hsi_masked_bam/channels_150' 'Data/hsi_masked_bam/channels_168')  # For maskedBAM
# data_dir=('Data/hsi_masked_bam_withoutTemp_LossMean/channels_25' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_50' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_75' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_100' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_125' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_150' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_168')
C=(50 75 100 125 150 168)
data_type=('maskedBAM')  # , 'maskedBAM' 'pcaLoading', 'maskedSPA'
model_name='densenet'  # Model name to be updated in the YAML file

yaml_file="models/hsi/config.yaml"  # Path to your YAML file

# Function to update the YAML file
update_yaml() {
    python -c "
import yaml

# Load YAML file
with open('$yaml_file', 'r') as file:
    config = yaml.safe_load(file)

# Update values
config['num_classes'] = $1
config['preprocessing'] = '$2'
config['data_dir'] = '$3'
config['C'] = $4
config['data_type'] = '$5'
config['model_name'] = '$6'  # Update model_name here

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3 $4 $5 "$6"
}

runner_file="models/hsi/trainer.py"  # Path to your Python script

# Iterate over each combination of num_classes, preprocessing, and zipped data_dir, C
for num_class in "${num_classes[@]}"; do
    for preprocessing in "${preprocessing[@]}"; do
        for idx in "${!data_dir[@]}"; do
            data="${data_dir[$idx]}"
            c_value="${C[$idx]}"
            for dtype in "${data_type[@]}"; do
                echo "Updating YAML and running $runner_file with num_classes=$num_class, preprocessing=$preprocessing, data_dir=$data, C=$c_value, data_type=$dtype, model_name=$model_name"

                # Update the YAML file with current parameters, including model_name
                update_yaml $num_class "$preprocessing" "$data" "$c_value" "$dtype" "$model_name"

                # Run the Python script and wait for it to complete
                python $runner_file
                if [ $? -ne 0 ]; then
                    echo "Error occurred while running the trainer with num_classes=$num_class, preprocessing=$preprocessing, data_dir=$data, C=$c_value, data_type=$dtype, model_name=$model_name"
                    exit 1
                fi

                echo "Completed num_classes=$num_class, preprocessing=$preprocessing, data_dir=$data, C=$c_value, data_type=$dtype, model_name=$model_name"
            done
        done
    done
done

echo "All combinations have been processed."
