#!/bin/bash

num_classes=(96)  # (12 24 37 55 75 96)
preprocessing=('none')
C=(25 50 75 100 125 150 168)

# Data directories for each data type
data_dir_maskedBAM=('Data/hsi_masked_bam/channels_25' 'Data/hsi_masked_bam/channels_50' 'Data/hsi_masked_bam/channels_75' 'Data/hsi_masked_bam/channels_100' 'Data/hsi_masked_bam/channels_125' 'Data/hsi_masked_bam/channels_150' 'Data/hsi_masked_bam/channels_168')  # For maskedBAM
# data_dir_maskedBAM=('Data/hsi_masked_bam_withoutTemp_LossMean/channels_25' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_50' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_75' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_100' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_125' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_150' 'Data/hsi_masked_bam_withoutTemp_LossMean/channels_168')  # For maskedBAM
data_dir_pcaLoading=('Data/hsi_masked_pcaLoading/Channel_25' 'Data/hsi_masked_pcaLoading/Channel_50' 'Data/hsi_masked_pcaLoading/Channel_75' 'Data/hsi_masked_pcaLoading/Channel_100' 'Data/hsi_masked_pcaLoading/Channel_125' 'Data/hsi_masked_pcaLoading/Channel_150' 'Data/hsi_masked_pcaLoading/Channel_168')  # For pcaLoading
data_dir_maskedSPA=('Data/hsi_masked_spa/Channel_25' 'Data/hsi_masked_spa/Channel_50' 'Data/hsi_masked_spa/Channel_75' 'Data/hsi_masked_spa/Channel_100' 'Data/hsi_masked_spa/Channel_125' 'Data/hsi_masked_spa/Channel_150' 'Data/hsi_masked_spa/Channel_168')  # For maskedSPA
model_name=('densenet')  # Model name
data_type=('maskedPCALoading')     #('maskedPCALoading' 'maskedSPA' 'maskedBAM')

hsi_yaml="models/hsi/config.yaml"
# runner_file="models/ensemble/base.py"
runner_file="models/ensemble/regression_ensemble.py"


# Function to update the HSI YAML file
update_hsi_yaml() {
    python -c "
import yaml
with open('$hsi_yaml', 'r') as file:
    config = yaml.safe_load(file)
config['num_classes'] = $1
config['preprocessing'] = '$2'
config['data_dir'] = '$3'
config['C'] = $4
config['data_type'] = '$5'
config['model_name'] = '$6'  # Add the model name
with open('$hsi_yaml', 'w') as file:
    yaml.safe_dump(config, file)
    " $1 "$2" "$3" $4 "$5" "$6"
}

# Iterate over each combination of parameters for HSI configuration
for num_class in "${num_classes[@]}"; do
    for preprocessing in "${preprocessing[@]}"; do
        for data_type_idx in "${!data_type[@]}"; do
            # Get the corresponding data_dir for this data_type
            if [ "${data_type[$data_type_idx]}" == "maskedBAM" ]; then
                data_dirs=("${data_dir_maskedBAM[@]}")
            elif [ "${data_type[$data_type_idx]}" == "maskedPCALoading" ]; then
                data_dirs=("${data_dir_pcaLoading[@]}")
            elif [ "${data_type[$data_type_idx]}" == "maskedSPA" ]; then
                data_dirs=("${data_dir_maskedSPA[@]}")
            fi

            # Zipping data_dir and C together
            for idx in "${!data_dirs[@]}"; do
                data_dir_value="${data_dirs[$idx]}"
                c_value="${C[$idx]}"

                echo "Updating HSI YAML and running $runner_file with num_classes=$num_class, data_dir=$data_dir_value, C=$c_value, data_type=${data_type[$data_type_idx]}, model_name=$model_name"

                # Update the HSI YAML file with current parameters
                update_hsi_yaml $num_class "$preprocessing" "$data_dir_value" "$c_value" "${data_type[$data_type_idx]}" "$model_name"

                # Run the Python script and check for errors
                python $runner_file
                if [ $? -ne 0 ]; then
                    echo "Error occurred while running the HSI trainer with num_classes=$num_class, data_dir=$data_dir_value, C=$c_value, data_type=${data_type[$data_type_idx]}, model_name=$model_name"
                    exit 1
                fi
                echo "Completed HSI num_classes=$num_class, data_dir=$data_dir_value, C=$c_value, data_type=${data_type[$data_type_idx]}, model_name=$model_name"
            done
        done
    done
done

echo "All combinations have been processed."
