
cd src/03_train_model
echo "starting training.sh"
echo $(pwd)
sh dvc_pull.sh -t $5
export GCP_PROJECT=$1
export GCS_BUCKET_NAME=$2
export WANDB_API_KEY=$4
python training.py -g $6
