#!/bin/bash

#====================== **WARNING** ===========================================
# Should be run in a VM or cloud instance (Ubuntu )
# This script will modify the configuration of the system and install things
# Should be run with no password sudo access
#==============================================================================

set -e

function main() {
    prepare_machine "$@"
    prepare_docker_image # Correspond docker-pytorch-linux-bionic-py3.6-clang9 CircleCI job
    build # Correspond pytorch_linux_bionic_py3_6_clang9_build CircleCI job
}

# Install dependencies and create circleci user
function prepare_machine() {
    if [ "$1" != "as-circleci" ] ; then

        # Installing Docker from official docker repos because is required for later
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl \
                                gnupg-agent software-properties-common
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo apt-key fingerprint 0EBFCD88
        sudo add-apt-repository \
            "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install -y docker-ce=5:18.09.4~3-0~ubuntu-xenial

        # Create circleci user with docker and sudo access
        sudo adduser circleci -q --disabled-password --gecos "" || true
        sudo adduser circleci docker
        sudo bash -c "echo \"circleci ALL=(ALL) NOPASSWD:ALL\" > /etc/sudoers.d/10-circleci-user"
        
        # Create link pip to pip3
        sudo ln -s /usr/bin/pip3 /usr/bin/pip
        
        # Continue as circleci user
        exec sudo su circleci -c "$(readlink -f $0) as-circleci"
    fi
    echo "Continuing as $(whoami)"
}

# Build Docker image
# See: https://app.circleci.com/pipelines/github/pytorch/pytorch/201040/workflows/05547b7e-f7c7-447e-b6cf-0158d15bc6e3/jobs/6745572
function prepare_docker_image() {
    export  CI=true \
            CIRCLECI=true \
            CIRCLE_BRANCH=master \
            CIRCLE_BUILD_NUM=6745572 \
            CIRCLE_BUILD_URL=https://circleci.com/gh/pytorch/pytorch/6745572  \
            CIRCLE_COMPARE_URL=  \
            CIRCLE_JOB=docker-pytorch-linux-bionic-py3.6-clang9 \
            CIRCLE_NODE_INDEX=0 \
            CIRCLE_NODE_TOTAL=1 \
            CIRCLE_PREVIOUS_BUILD_NUM=6745571 \
            CIRCLE_PROJECT_REPONAME=pytorch \
            CIRCLE_PROJECT_USERNAME=pytorch \
            CIRCLE_REPOSITORY_URL=https://github.com/pytorch/pytorch \
            CIRCLE_SHA1=248b6a30f4d5b08876d1e7e6f350875ff6c1c5da \
            CIRCLE_SHELL_ENV=/tmp/.bash_env-5f3a300e43589b612a80a17b-0-build \
            CIRCLE_STAGE=docker-pytorch-linux-bionic-py3.6-clang9 \
            CIRCLE_USERNAME=facebook-github-bot \
            CIRCLE_WORKFLOW_ID=05547b7e-f7c7-447e-b6cf-0158d15bc6e3 \
            CIRCLE_WORKFLOW_JOB_ID=5159d8c6-9ece-4d27-82cd-3bfa0dc51fb3 \
            CIRCLE_WORKFLOW_UPSTREAM_JOB_IDS= \
            CIRCLE_WORKFLOW_WORKSPACE_ID=05547b7e-f7c7-447e-b6cf-0158d15bc6e3 \
            CIRCLE_WORKING_DIRECTORY=~/project

    mkdir -p /home/circleci/project
    cd /home/circleci/project
    git clone https://github.com/pytorch/pytorch .

    export DOCKER_TAG=$(git rev-parse HEAD:.circleci/docker)
    export IMAGE_NAME=pytorch-linux-bionic-py3.6-clang9
    export DOCKER_CLI_EXPERIMENTAL="enabled"
    export DOCKER_BUILDKIT="1"

    # Original command: cd .circleci/docker && ./build_docker.sh
    # this script deals with AWS to check if the docker image should be build 
    # and to push it to AWS ECR once built
    # We can summarize the script to:
    cd .circleci/docker
    ./build.sh $IMAGE_NAME -t $IMAGE_NAME:$DOCKER_TAG
    cd ../..
}

# 
function build() {
    export BASH_ENV=/tmp/.bash_env-5f3a30442147c967bb3294a3-0-build \
           CI=true \
           CIRCLECI=true \
           CIRCLE_BRANCH=master \
           CIRCLE_BUILD_NUM=6745583 \
           CIRCLE_BUILD_URL=https://circleci.com/gh/pytorch/pytorch/6745583 \
           CIRCLE_COMPARE_URL= \
           CIRCLE_JOB=pytorch_linux_bionic_py3_6_clang9_build \
           CIRCLE_NODE_INDEX=0 \
           CIRCLE_NODE_TOTAL=1 \
           CIRCLE_PREVIOUS_BUILD_NUM=6745582 \
           CIRCLE_PROJECT_REPONAME=pytorch \
           CIRCLE_PROJECT_USERNAME=pytorch \
           CIRCLE_REPOSITORY_URL=https://github.com/pytorch/pytorch \
           CIRCLE_SHA1=248b6a30f4d5b08876d1e7e6f350875ff6c1c5da \
           CIRCLE_SHELL_ENV=/tmp/.bash_env-5f3a30442147c967bb3294a3-0-build \
           CIRCLE_STAGE=pytorch_linux_bionic_py3_6_clang9_build \
           CIRCLE_USERNAME=facebook-github-bot \
           CIRCLE_WORKFLOW_ID=05547b7e-f7c7-447e-b6cf-0158d15bc6e3 \
           CIRCLE_WORKFLOW_JOB_ID=3099fd50-d6f3-445f-81b4-a60d42f1d6a9 \
           CIRCLE_WORKFLOW_UPSTREAM_JOB_IDS=5159d8c6-9ece-4d27-82cd-3bfa0dc51fb3 \
           CIRCLE_WORKFLOW_WORKSPACE_ID=05547b7e-f7c7-447e-b6cf-0158d15bc6e3 \
           CIRCLE_WORKING_DIRECTORY=~/project
    
    # Skip checkout of code since already done in previous "CircleCI Job"
    
    # See pytorch_params section of CircleCI config
    export BUILD_ENVIRONMENT="pytorch-linux-bionic-py3.6-clang9-build"
    export DOCKER_IMAGE=$IMAGE_NAME # No access to aws ecr, use local image
    export USE_CUDA_RUNTIME=""
    export BUILD_ONLY=""
    
    .circleci/scripts/setup_linux_system_environment.sh
    .circleci/scripts/setup_ci_environment.sh
    
    # Run build
    echo "DOCKER_IMAGE: "${DOCKER_IMAGE}:${DOCKER_TAG}
    export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${DOCKER_IMAGE}:${DOCKER_TAG})

    git submodule sync && git submodule update -q --init --recursive

    docker cp /home/circleci/project/. $id:/var/lib/jenkins/workspace

    if [[ ${BUILD_ENVIRONMENT} == *"paralleltbb"* ]]; then
        export PARALLEL_FLAGS="export ATEN_THREADING=TBB USE_TBB=1 "
    elif [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
        export PARALLEL_FLAGS="export ATEN_THREADING=NATIVE "
    fi
    echo "Parallel backend flags: "${PARALLEL_FLAGS}

    export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo '"$PARALLEL_FLAGS"' && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/build.sh && find ${BUILD_ROOT} -type f -name "*.a" -or -name "*.o" -delete") | docker exec -u jenkins -i "$id" bash) 2>&1'

    echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

    # Push intermediate Docker image for next phase to use
    COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-built
    docker commit "$id" ${COMMIT_DOCKER_IMAGE}
}

main "$@"
