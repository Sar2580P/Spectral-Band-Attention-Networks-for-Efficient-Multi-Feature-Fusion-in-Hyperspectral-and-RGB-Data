#!/bin/bash

# Define the range of folds and the list of num_classes
folds=(0 1 2 3 4)
num_classes=(12 24 37 55 75 96)
yaml_file="models/hsi/dense_net/config.yaml"  # Path to your YAML file

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

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2
}

runner_file="models/hsi/dense_net/denseNet.py"  # Path to your Python script

# Iterate over each combination of fold and num_classes
for fold in "${folds[@]}"; do
    for num_class in "${num_classes[@]}"; do
        echo "Updating YAML and running $runner_file with fold=$fold and num_classes=$num_class"

        # Update the YAML file with current fold and num_class
        update_yaml $fold $num_class

        # Run the Python script
        python $runner_file &

        # Wait for the previous run to complete
        wait

        echo "Completed fold=$fold and num_classes=$num_class"
    done
done

echo "All combinations have been processed."
