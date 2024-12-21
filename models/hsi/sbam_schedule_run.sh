#!/bin/bash

# Define the list of parameters you want to loop over
num_classes=(96)  # Modify this to include other values like (12, 24, 37) if needed
preprocessing=('none')  # Modify to include other preprocessing methods if needed
data_dir=('Data/hsi_masked')  # Modify to include other directories as needed
C=(168)  # Modify this to include other values if needed
data_type=('maskedPCALoading')  # Modify if you need different data types
model_name='sparse_bam_densenet_withoutTemp_LossMean'  # New model_name to be updated in the config

yaml_file="models/hsi/config.yaml"  # Path to your YAML file

# Function to update the YAML file dynamically, now including model_name
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
config['model_name'] = '$6'  # Add model_name update here

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3 $4 $5 "$6"
}

runner_file="models/hsi/trainer.py"  # Path to your Python script

# Iterate over each combination of num_classes, preprocessing, data_dir, C, and model_name
for num_class in "${num_classes[@]}"; do
    for preprocessing in "${preprocessing[@]}"; do
        for i in "${!data_dir[@]}"; do
            for c_value in "${C[@]}"; do
                for dtype in "${data_type[@]}"; do
                    echo "Updating YAML and running $runner_file with num_classes=$num_class, preprocessing=$preprocessing, data_dir=${data_dir[$i]}, C=$c_value, data_type=$dtype, model_name=$model_name"

                    # Update the YAML file with current parameters, including model_name
                    update_yaml $num_class "$preprocessing" "${data_dir[$i]}" "$c_value" "$dtype" "$model_name"

                    # Run the Python script and wait for it to complete
                    python $runner_file
                    if [ $? -ne 0 ]; then
                        echo "Error occurred while running the trainer with num_classes=$num_class, preprocessing=$preprocessing, data_dir=${data_dir[$i]}, C=$c_value, data_type=$dtype, model_name=$model_name"
                        exit 1
                    fi

                    echo "Completed num_classes=$num_class, preprocessing=$preprocessing, data_dir=${data_dir[$i]}, C=$c_value, data_type=$dtype, model_name=$model_name"
                done
            done
        done
    done
done

echo "All combinations have been processed."
