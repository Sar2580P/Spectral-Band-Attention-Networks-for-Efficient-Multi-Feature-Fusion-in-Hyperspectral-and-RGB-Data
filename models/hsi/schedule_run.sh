#!/bin/bash

# Define the range of folds and the list of num_classes
folds=(4)      #(0 1 2 3 4)
num_classes=(96)   #(12 24 37 55 75 96)
preprocessing=('msc' 'snv' 'none')
data_dir=('Data/hsi_pca' 'Data/hsi_trimmed' 'Data/hsi')
C=(30 158 168)
data_type=('pca' 'trimmed' 'original')
apply_bam=('True' 'False')
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
config['apply_bam'] = $5
config['C'] = $6
config['data_type'] = '$7'

# Save updated YAML file
with open('$yaml_file', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 $2 $3 $4 $5 $6 $7
}

runner_file="models/hsi/trainer.py"  # Path to your Python script

# Iterate over each combination of fold and num_classes
for fold in "${folds[@]}"; do
    for num_class in "${num_classes[@]}"; do
        for preprocessing in "${preprocessing[@]}"; do
            for i in "${!data_dir[@]}"; do
                for apply_bam in "${apply_bam[@]}"; do
                    echo "Updating YAML and running $runner_file with fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[$i]}"

                    # Update the YAML file with current parameters
                    update_yaml $fold $num_class "$preprocessing" "${data_dir[$i]}" "$apply_bam" "${C[$i]}" "${data_type[$i]}"

                    # Run the Python script
                    python $runner_file &

                    # Wait for the previous run to complete
                    wait

                    echo "Completed fold=$fold, num_classes=$num_class, data_dir=${data_dir[$i]}, C=${C[$i]}, data_type=${data_type[$i]}"
                done
            done
        done
    done
done

echo "All combinations have been processed."
