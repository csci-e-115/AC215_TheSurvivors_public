
while getopts "t:" opt
do
   case "$opt" in
      t) TAG="$OPTARG" ;;
   esac
done

# cp ../02_preprocess_data/dataset_paths.dvc . && \
# dvc pull

echo "starting dvc pull"
echo "git url = $GIT_URL"
echo "tag = $TAG"
dvc get $GIT_URL Versioning --force --rev $TAG
cat Versioning/versioned_paths.pickle
echo "finished dvc"