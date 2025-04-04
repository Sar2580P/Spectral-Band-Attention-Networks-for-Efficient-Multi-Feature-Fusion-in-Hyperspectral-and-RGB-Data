#!/bin/bash

# Define the list of num_classes
num_classes=(96)
preprocessing=('none')
C=(25 50 75 100 125 150)
data_type=('TripleAttnDataset')  # , 'maskedBAM' 'pcaLoading', 'maskedSPA'
model_name='densenet'  # Model name to be updated in the YAML file

yaml_file="models/hsi/config.yaml"  # Path to your YAML file

# Function to update only specified parameters in the YAML file
update_yaml() {
    python -c "
import yaml

# Load YAML file
with open('$yaml_file', 'r') as file:
    config = yaml.safe_load(file)

# Ensure config is not None
if config is None:
    config = {}

# Update only specified parameters
config.update({
    'num_classes': int($1),
    'preprocessing': '$2',
    'C': int($3),
    'data_type': '$4',
    'model_name': '$5'
})

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file, default_flow_style=False)
" $1 "$2" $3 "$4" "$5"
}

runner_file="models/ensemble/regression_ensemble.py"  # Path to your Python script

# Iterate over each combination of num_classes, preprocessing, and C
for num_class in "${num_classes[@]}"; do
    for preprocessing in "${preprocessing[@]}"; do
        for idx in "${!C[@]}"; do
            c_value="${C[$idx]}"
            for dtype in "${data_type[@]}"; do
                echo "Updating YAML and running $runner_file with num_classes=$num_class, preprocessing=$preprocessing, C=$c_value, data_type=$dtype, model_name=$model_name"

                # Update the YAML file with current parameters
                update_yaml $num_class "$preprocessing" "$c_value" "$dtype" "$model_name"

                # Run the Python script and wait for it to complete
                python $runner_file
                if [ $? -ne 0 ]; then
                    echo "Error occurred while running the trainer with num_classes=$num_class, preprocessing=$preprocessing, C=$c_value, data_type=$dtype, model_name=$model_name"
                    exit 1
                fi

                echo "Completed num_classes=$num_class, preprocessing=$preprocessing, C=$c_value, data_type=$dtype, model_name=$model_name"
            done
        done
    done
done

echo "All combinations have been processed."
