#!/bin/bash

#====================== **WARNING** ===========================================
# Should be run in a VM or cloud instance (used c5d.2xlarge with Ubuntu 16.04)
# This script will modify the configuration of the system and install things
# Should be run with no password sudo access
#==============================================================================

set -e

function main() {
    prepare_machine "$@"
    prepare_docker_image # Correspond docker-pytorch-linux-bionic-py3.6-clang9 CircleCI job
    build # Correspond pytorch_linux_bionic_py3_6_clang9_build CircleCI job
    sudo poweroff
}

# Install dependencies and create circleci user
function prepare_machine() {
    if [ "$1" != "as-circleci" ] ; then

        # Installing Docker from official docker repos because is required for later
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl \
                                gnupg-agent software-properties-common jq
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
    git checkout 490d41aaa61a9c0b12637e40cec066bf0e9515f3 # patchs regularly get broken
    
    # Add verificarlo to docker image
    patch_docker_build_scripts

    export DOCKER_TAG=$(git rev-parse HEAD:.circleci/docker)
    # export IMAGE_NAME=pytorch-linux-bionic-py3.6-clang9
    export IMAGE_NAME=pytorch-linux-bionic-py3.6-verificarlo
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

function patch_docker_build_scripts() {
cat << 'EOF' | git apply
diff --git a/.circleci/docker/build.sh b/.circleci/docker/build.sh
index 20a80cf..76524fc 100755
--- a/.circleci/docker/build.sh
+++ b/.circleci/docker/build.sh
@@ -202,8 +202,17 @@ case "$image" in
     DB=yes
     VISION=yes
     VULKAN_SDK_VERSION=1.2.148.0
     SWIFTSHADER=yes
     ;;
+  pytorch-linux-bionic-py3.6-verificarlo)
+    ANACONDA_PYTHON_VERSION=3.6
+    CLANG_VERSION=9
+    VERIFICARLO_VERSION=github
+    LLVMDEV=yes
+    PROTOBUF=yes
+    DB=yes
+    VISION=yes
+    ;;
   pytorch-linux-bionic-py3.8-gcc9)
     ANACONDA_PYTHON_VERSION=3.8
     GCC_VERSION=9
@@ -345,6 +354,7 @@ docker build \
        --build-arg "DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}" \
        --build-arg "GLIBC_VERSION=${GLIBC_VERSION}" \
        --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
+       --build-arg "VERIFICARLO_VERSION=${VERIFICARLO_VERSION}" \
        --build-arg "ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION}" \
        --build-arg "TRAVIS_PYTHON_VERSION=${TRAVIS_PYTHON_VERSION}" \
        --build-arg "GCC_VERSION=${GCC_VERSION}" \
diff --git a/.circleci/docker/common/install_verificarlo.sh b/.circleci/docker/common/install_verificarlo.sh
new file mode 100755
index 0000000..afd1342
--- /dev/null
+++ b/.circleci/docker/common/install_verificarlo.sh
@@ -0,0 +1,24 @@
+#!/bin/bash
+
+set -ex
+
+if [ -n "$VERIFICARLO_VERSION" ]; then
+  sudo apt-get update
+  sudo apt-get install -y --no-install-recommends libmpfr-dev libtool
+  pip install bigfloat
+
+  git clone https://github.com/verificarlo/verificarlo
+  cd verificarlo
+  git checkout v0.4.0
+  export PATH="$PATH:/usr/lib/llvm-9/bin/"
+  ./autogen.sh
+  ./configure --without-flang CC=gcc-7 CXX=g++-7
+  make
+  sudo make install
+  cd ..
+  rm -rf verificarlo
+
+  # Cleanup package manager
+  apt-get autoclean && apt-get clean
+  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
+fi
diff --git a/.circleci/docker/ubuntu/Dockerfile b/.circleci/docker/ubuntu/Dockerfile
index 61d426f..a0dcc79 100644
--- a/.circleci/docker/ubuntu/Dockerfile
+++ b/.circleci/docker/ubuntu/Dockerfile
@@ -17,6 +17,11 @@ ARG CLANG_VERSION
 ADD ./common/install_travis_python.sh install_travis_python.sh
 RUN bash ./install_travis_python.sh && rm install_travis_python.sh
 
+# Install Verificarlo
+ARG VERIFICARLO_VERSION
+ADD ./common/install_verificarlo.sh install_verificarlo.sh
+RUN bash ./install_verificarlo.sh && rm install_verificarlo.sh
+
 # (optional) Install protobuf for ONNX
 ARG PROTOBUF
 ADD ./common/install_protobuf.sh install_protobuf.sh
-- 
2.7.4

EOF
}

function patch_build_script_to_handle_verificarlo() {
cat << 'EOF' | git apply
diff --git a/.jenkins/pytorch/build.sh b/.jenkins/pytorch/build.sh
index 8730b61..23c9f92 100755
--- a/.jenkins/pytorch/build.sh
+++ b/.jenkins/pytorch/build.sh
@@ -194,6 +194,12 @@ if [[ "${BUILD_ENVIRONMENT}" == *clang* ]]; then
   export CXX=clang++
 fi

+if [[ "${BUILD_ENVIRONMENT}" == *verificarlo* ]]; then
+  export CC=verificarlo-c
+  export CXX=verificarlo-c++
+  export VFC_BACKENDS=libinterflop_ieee.so # So that cmake checks pass
+fi
+
 # Patch required to build xla
 if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
   git clone --recursive https://github.com/pytorch/xla.git
@@ -218,7 +224,7 @@ else
     # ppc64le build fails when WERROR=1
     # set only when building other architectures
     # only use for "python setup.py install" line
-    if [[ "$BUILD_ENVIRONMENT" != *ppc64le*  && "$BUILD_ENVIRONMENT" != *clang* ]]; then
+    if [[ "$BUILD_ENVIRONMENT" != *ppc64le*  && "$BUILD_ENVIRONMENT" != *clang* && "$BUILD_ENVIRONMENT" != *verificarlo* ]]; then
       WERROR=1 python setup.py install
     else
       python setup.py install
diff --git a/.jenkins/pytorch/common_utils.sh b/.jenkins/pytorch/common_utils.sh
index 682dd29b4c..645063251b 100644
--- a/.jenkins/pytorch/common_utils.sh
+++ b/.jenkins/pytorch/common_utils.sh
@@ -18,7 +18,7 @@ function cleanup {
 function assert_git_not_dirty() {
     # TODO: we should add an option to `build_amd.py` that reverts the repo to
     #       an unmodified state.
-    if ([[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *xla* ]]) ; then
+    if ([[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *xla* ]] && [[ "$BUILD_ENVIRONMENT" != *verificarlo* ]]) ; then
         git_status=$(git status --porcelain)
         if [[ $git_status ]]; then
             echo "Build left local git repository checkout dirty"
EOF
}

function setup_function_instrumentation() {
# Also adding -Qunused-arguments arg to fix Unused arguement warnings (becomes error with -Werror)
id=$1
cat << 'EOFF' | docker exec -u jenkins -i "$id" bash 
sudo mkdir -p /etc/verificarlo
cat << 'EOF' | sudo tee /etc/verificarlo/inclusion-file >> /dev/null
# [file without suffix] [mangled function name]
# Calls to verificarlo are made using full paths
/var/lib/jenkins/workspace/build/aten/src/ATen/native/cpu/BlasKernel.cpp.DEFAULT *
/var/lib/jenkins/workspace/build/aten/src/ATen/native/cpu/BlasKernel.cpp.AVX *
/var/lib/jenkins/workspace/build/aten/src/ATen/native/cpu/BlasKernel.cpp.AVX2 *
EOF
cat << 'EOF' | sudo tee /usr/local/bin/verificarlo-c >> /dev/null
#!/bin/bash

verificarlo --linker=clang -Qunused-arguments --include-file=/etc/verificarlo/inclusion-file ${@}
EOF
cat << 'EOF' | sudo tee /usr/local/bin/verificarlo-c++ >> /dev/null
#!/bin/bash

verificarlo --linker=clang++ -Qunused-arguments --include-file=/etc/verificarlo/inclusion-file ${@}
EOF
EOFF
}

function disable_blas() {
cat << 'EOF' | git apply
diff --git a/.jenkins/pytorch/build.sh b/.jenkins/pytorch/build.sh
index 23c9f92..f972150 100755
--- a/.jenkins/pytorch/build.sh
+++ b/.jenkins/pytorch/build.sh
@@ -227,6 +227,8 @@ else
     if [[ "$BUILD_ENVIRONMENT" != *ppc64le*  && "$BUILD_ENVIRONMENT" != *clang* && "$BUILD_ENVIRONMENT" != *verificarlo* ]]; then
       WERROR=1 python setup.py install
     else
+      python setup.py build --cmake-only
+      sed -i "s/AT_BUILD_WITH_BLAS() 1/AT_BUILD_WITH_BLAS() 0/" aten/src/ATen/Config.h
       python setup.py install
     fi

EOF
}

function disable_parallel_compilation() {
sed -Ei "s/MAX_JOBS=[0-9]+/MAX_JOBS=1/" $BASH_ENV
}

function build() {
    export BASH_ENV=/home/circleci/project/env \
           CI=true \
           CIRCLECI=true \
           CIRCLE_BRANCH=master \
           CIRCLE_COMPARE_URL= \
           CIRCLE_JOB=pytorch_linux_bionic_py3_6_clang9_build \
           CIRCLE_NODE_INDEX=0 \
           CIRCLE_NODE_TOTAL=1 \
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
           CIRCLE_WORKING_DIRECTORY=~/project \
    
    # Skip checkout of code since already done in previous "CircleCI Job"
    
    # See pytorch_params section of CircleCI config
    # export BUILD_ENVIRONMENT="pytorch-linux-bionic-py3.6-clang9-build"
    export BUILD_ENVIRONMENT="pytorch-linux-bionic-py3.6-verificarlo-build"
    export DOCKER_IMAGE=$IMAGE_NAME # No access to aws ecr, use local image
    export USE_CUDA_RUNTIME=""
    export BUILD_ONLY=""
    
    .circleci/scripts/setup_linux_system_environment.sh
    .circleci/scripts/setup_ci_environment.sh

    echo "Launching the build docker container(${DOCKER_IMAGE}:${DOCKER_TAG})"
    export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${DOCKER_IMAGE}:${DOCKER_TAG})
    git submodule sync && git submodule update -q --init --recursive

    # Customizations:
    patch_build_script_to_handle_verificarlo
    setup_function_instrumentation $id
    disable_parallel_compilation
    disable_blas

    docker cp /home/circleci/project/. $id:/var/lib/jenkins/workspace

    if [[ ${BUILD_ENVIRONMENT} == *"paralleltbb"* ]]; then
        export PARALLEL_FLAGS="export ATEN_THREADING=TBB USE_TBB=1 "
    elif [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
        export PARALLEL_FLAGS="export ATEN_THREADING=NATIVE "
    fi
    echo "Parallel backend flags: "${PARALLEL_FLAGS}

    export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo '"$PARALLEL_FLAGS"' && echo "set -a && source ./workspace/env && set +a" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/build.sh && find ${BUILD_ROOT} -type f -name "*.a" -or -name "*.o" -or -name "*.ll" -delete") | docker exec -u jenkins -i "$id" bash) 2>&1'

    echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

    # Commit built Docker image
    COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-built
    docker commit "$id" ${COMMIT_DOCKER_IMAGE}
    
    test_fuzzy_pytorch ${COMMIT_DOCKER_IMAGE}
    if [ "$?" -eq "0" ] ; then # Tests passed
        DOCKERHUB_TOKEN=$(aws secretsmanager get-secret-value --region us-east-2 --secret-id DockerToken | jq -r '.SecretString')
        echo $DOCKERHUB_TOKEN | docker login --username hantoine --password-stdin
        docker tag ${COMMIT_DOCKER_IMAGE} hantoine/fuzzy-pytorch
        docker push hantoine/fuzzy-pytorch
    fi
}

function test_fuzzy_pytorch() {
docker_image=$1
cat << 'EOF' | docker run -i $docker_image bash
VFC_BACKENDS="libinterflop_mca.so" python -c "
import torch
import pickle

N_SAMPLES = 100

a = torch.tensor([[0.5885, 0.6663, 0.4825, 0.4913, 0.3757],
                  [0.3911, 0.7989, 0.3581, 0.3432, 0.7236],
                  [0.5128, 0.2178, 0.2068, 0.7906, 0.5767],
                  [0.0037, 0.2509, 0.2034, 0.8508, 0.5715],
                  [0.1072, 0.4296, 0.3166, 0.3436, 0.2760]])
b = torch.tensor([[0.8158, 0.1416, 0.1986, 0.9440, 0.1942],
                  [0.8634, 0.9054, 0.4473, 0.6808, 0.3795],
                  [0.2371, 0.1120, 0.4089, 0.9029, 0.1519],
                  [0.4150, 0.6956, 0.6720, 0.1385, 0.8701],
                  [0.6150, 0.1443, 0.0611, 0.4893, 0.1078]])

res = []
for _ in range(N_SAMPLES):
  res.append(a @ b)

with open('fuzzy-pytorch_mean_result.pickle', 'wb') as file:
  pickle.dump(torch.stack(res).mean(dim=0), file)
"
VFC_BACKENDS="libinterflop_ieee.so" python -c "
import torch
import pickle

a = torch.tensor([[0.5885, 0.6663, 0.4825, 0.4913, 0.3757],
                  [0.3911, 0.7989, 0.3581, 0.3432, 0.7236],
                  [0.5128, 0.2178, 0.2068, 0.7906, 0.5767],
                  [0.0037, 0.2509, 0.2034, 0.8508, 0.5715],
                  [0.1072, 0.4296, 0.3166, 0.3436, 0.2760]])
b = torch.tensor([[0.8158, 0.1416, 0.1986, 0.9440, 0.1942],
                  [0.8634, 0.9054, 0.4473, 0.6808, 0.3795],
                  [0.2371, 0.1120, 0.4089, 0.9029, 0.1519],
                  [0.4150, 0.6956, 0.6720, 0.1385, 0.8701],
                  [0.6150, 0.1443, 0.0611, 0.4893, 0.1078]])

res = a @ b

with open('fuzzy-pytorch_mean_result.pickle', 'rb') as file:
  mean_res = pickle.load(file)

relative_errors = (mean_res - res) / res
print('Relative errors: ')
print(relative_errors)

assert torch.allclose(mean_res, res)
print('Test of matrix multiplication passed')
"
EOF
return $?
}

main "$@"
