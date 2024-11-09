echo "Jai Shree Ram"
user_name=$(whoami)
path_to_env="/home/${user_name}/.cache/pypoetry/virtualenvs/wheat-seed-classification-*"
if [ -d $path_to_env ]; then
  echo "Found the existing environment!!"
else
  echo "Creating a working environment ..."
  poetry install --no-root

fi
echo -e "\n\nActivating the working environment..."


source .env
wandb login --relogin "$WANDB_API"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

chmod +x models/hsi/schedule_run.sh
chmod +x models/rgb/schedule_run.sh

poetry shell
