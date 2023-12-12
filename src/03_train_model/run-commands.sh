# $1 GCP_PROJECT
# $2 GCS_BUCKET_NAME
# $3 GIT_LOGIN
# $4 WANDB_API_KEY
# $5 DVC_TAG
# $6 NUM_GPU

echo "\$1 = $1"
echo "\$2 = $2"
echo "\$3 = $3"
echo "\$4 = $4"
echo "\$5 = $5"
echo "\$6 = $6"
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_USEREMAIL"
git config --global user.password "$GIT_PASSWORD"

git init
git remote add -f origin $3@github.com/csci-e-115/AC215_TheSurvivors.git
git sparse-checkout init
git sparse-checkout set src
git pull origin milestone4
export GIT_URL=$3@github.com/csci-e-115/dermaid-versioning
echo $GIT_URL
git submodule init
git config submodule.src/08_versioning.url $GIT_URL
git submodule update --init --recursive

pipenv run bash run-training.sh $1 $2 $3 $4 $5 $6