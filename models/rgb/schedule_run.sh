#!/bin/bash

# Define the range of folds and the list of num_classes
folds=(0 1 2 3 4)
num_classes=(96)  # New requirement only for num_classes=96

# Define the list of runner files and their corresponding config files
runner_files=("models/rgb/trainer.py" )
config_files=("models/rgb/config.yaml")
model_names=("densenet" "resnet" "google_net")


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
config['model_name'] = '$4'

# Save updated YAML file
with open('$1', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3 "$4"
}

# Set the working directory if necessary
cd /home/bala/Desktop/sri_krishna/Wheat-Seed-Classification || exit

# Iterate over each combination of fold and num_classes
for fold in "${folds[@]}"; do
    for num_class in "${num_classes[@]}"; do
        for index in "${!runner_files[@]}"; do
            for model_name in "${model_names[@]}"; do
                runner_file=${runner_files[$index]}
                config_file=${config_files[$index]}

                echo "Updating $config_file and dataloading.py and running $runner_file with fold=$fold and num_classes=$num_class and model_name=$model_name"

                # Update the YAML config file with current fold, num_class, and model_name
                update_yaml $config_file $fold $num_class "$model_name"

                # Run the Python script
                PYTHONPATH=$(pwd) python $runner_file &

                # Wait for the previous run to complete
                wait

                echo "Completed $runner_file with fold=$fold and num_classes=$num_class and model_name=$model_name"
            done
        done
    done
done

echo "All combinations have been processed."
