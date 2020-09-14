#!/bin/bash

#====================== **WARNING** ===========================================
# Should be run in a VM or cloud instance (used c5d.2xlarge with Ubuntu 16.04)
# This script will modify the configuration of the system and install things
# Should be run as root
#==============================================================================

DOCKERHUB_USER="hantoine"

set -e

function main() {
    prepare_machine

    git clone https://github.com/hantoine/fuzzy-networks
    cd fuzzy-networks/pytorch/build

    build_buildenv_image
    build_built_image
    build_final_image
    build_final_jupyter_image

    echo "==========================="
    echo "  BUILD SUCCEEDED"
    echo "==========================="
    poweroff
}

# Install dependencies and create circleci user
function prepare_machine() {
    # Installing docker
    apt-get install -y docker.io

    # Logging in to Docker Hub
    aws secretsmanager get-secret-value --region us-east-2 --secret-id DockerToken --output text \
        | head -n 1 | cut  -f 4 \
        | docker login --username $DOCKERHUB_USER --password-stdin
}

function build_buildenv_image() {
    docker_tag=$(git rev-parse HEAD:pytorch/build/buildenv)
    buildenv_img="$DOCKERHUB_USER/fuzzy-pytorch-buildenv:$docker_tag"

    set +e
    docker pull $buildenv_img
    docker_pull_ret=$?
    set -e
    if [ "$docker_pull_ret" -ne "0" ] ; then

        docker build -t $buildenv_img buildenv
        docker push $buildenv_img
    fi
    docker tag $buildenv_img fuzzy-pytorch-buildenv
}

function build_built_image() {
    docker build --build-arg MAX_JOBS=$(($(nproc) - 1)) -t fuzzy-pytorch-built built
}

function build_final_image() {
    docker build -t $DOCKERHUB_USER/fuzzy-pytorch final
    cat test_fuzzy_pytorch.sh | docker run -i $DOCKERHUB_USER/fuzzy-pytorch
    if [ "$?" -ne "0" ] ; then # Tests failed
        poweroff
        exit 1
    fi
    docker push $DOCKERHUB_USER/fuzzy-pytorch
}

build_final_jupyter_image() {
    docker build final_jupyter -t $DOCKERHUB_USER/fuzzy-pytorch:jupyter
    docker push $DOCKERHUB_USER/fuzzy-pytorch:jupyter
}

main "$@"
