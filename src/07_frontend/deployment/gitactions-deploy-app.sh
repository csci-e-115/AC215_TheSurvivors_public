echo "Starting app deployment shell process"
ansible-playbook deploy-docker-images.yml -i inventory.yml
echo "Completed deploying docker images to gcr"
# ansible-playbook deploy-setup-containers.yml -i inventory.yml
# echo "Completed set up of containers on VM"
# ansible-playbook deploy-setup-webserver.yml -i inventory.yml
# echo "Completed deployed nginx server"
ansible-playbook deploy-update-k8s-cluster.yml -i inventory.yml