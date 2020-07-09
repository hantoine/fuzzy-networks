#!/bin/bash

#====================== **WARNING** ===========================================
# Should be run in a VM or cloud instance
# This script will modify the configuration of the system and install things
# Should be run as root
#==============================================================================

# CircleCI jobs uses a machine executor with ubuntu-1604:201903-01 image

# pytorch_linux_build Job main step
# Environement variables expected:
#     BUILD_ENVIRONMENT
#     DOCKER_IMAGE
#     USE_CUDA_DOCKER_RUNTIME
#     BUILD_ONLY
function pytorch_linux_build() {
    set -e
    # Pull Docker image and run build
    echo "DOCKER_IMAGE: "${DOCKER_IMAGE}
    time docker pull ${DOCKER_IMAGE} >/dev/null
    export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})
    git submodule sync && git submodule update -q --init --recursive
    docker cp /home/circleci/project/. $id:/var/lib/jenkins/workspace
    if [[ ${BUILD_ENVIRONMENT} == *"paralleltbb"* ]]; then
    export PARALLEL_FLAGS="export ATEN_THREADING=TBB USE_TBB=1 "
    elif [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
    export PARALLEL_FLAGS="export ATEN_THREADING=NATIVE "
    fi
    echo "Parallel backend flags: "${PARALLEL_FLAGS}
    export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo '"$PARALLEL_FLAGS"' && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/build.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
    echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts
    # Push intermediate Docker image for next phase to use
    if [ -z "${BUILD_ONLY}" ]; then
    # Note [Special build images]
    # The xla build uses the same docker image as
    # pytorch-linux-trusty-py3.6-gcc5.4-build. In the push step, we have to
    # distinguish between them so the test can pick up the correct image.
    output_image=${DOCKER_IMAGE}-${CIRCLE_SHA1}
    if [[ ${BUILD_ENVIRONMENT} == *"xla"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-xla
    elif [[ ${BUILD_ENVIRONMENT} == *"libtorch"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-libtorch
    elif [[ ${BUILD_ENVIRONMENT} == *"paralleltbb"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-paralleltbb
    elif [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-parallelnative
    elif [[ ${BUILD_ENVIRONMENT} == *"android-ndk-r19c-x86_64"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-android-x86_64
    elif [[ ${BUILD_ENVIRONMENT} == *"android-ndk-r19c-arm-v7a"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-android-arm-v7a
    elif [[ ${BUILD_ENVIRONMENT} == *"android-ndk-r19c-arm-v8a"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-android-arm-v8a
    elif [[ ${BUILD_ENVIRONMENT} == *"android-ndk-r19c-x86_32"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-android-x86_32
    elif [[ ${BUILD_ENVIRONMENT} == *"android-ndk-r19c-vulkan-x86_32"* ]]; then
        export COMMIT_DOCKER_IMAGE=$output_image-android-vulkan-x86_32
    else
        export COMMIT_DOCKER_IMAGE=$output_image
    fi
    docker commit "$id" ${COMMIT_DOCKER_IMAGE}
    time docker push ${COMMIT_DOCKER_IMAGE}
    fi
}

# In the CircleCI job, the Docker image is pulled from ECR, but it is not 
# accessible. The image  is built for each environment by the job docker_build_job
# Luckily most of ths job correspond to one script only, the rest is in charge 
# of checking when the image should be rebuilt and pushing it once it's built 
function prepare_docker_image() {
    cd .circleci/docker
    ./build.sh pytorch-linux-bionic-py3.6-clang9 -t hantoine/pytorch-linux-bionic-py3.6-clang9
    cd ../..
}

apt-get update
apt-get install -y moreutils expect
git clone https://github.com/pytorch/pytorch project
cd project
./.circleci/scripts/setup_linux_system_environment.sh
./.circleci/scripts/setup_ci_environment.sh

# Skipping prepare_docker_image, the docker image is already built and pushed to Docker Hub
# prepare_docker_image
adduser circleci -q --disabled-password --gecos ""
adduser circleci docker
exec sudo su circleci

# Corresponds to Job instanciation named pytorch_linux_bionic_py3_6_clang9_build
export BUILD_ENVIRONMENT="pytorch-linux-bionic-py3.6-clang9-build"
export DOCKER_IMAGE="hantoine/pytorch-linux-bionic-py3.6-clang9" # Use our image
export USE_CUDA_DOCKER_RUNTIME=""
export BUILD_ONLY=""
# Excpect pytorch in /home/circleci/project
pytorch_linux_build
