#!/bin/bash

#====================== **WARNING** ===========================================
# Should be run in a VM or cloud instance (used c5d.2xlarge with Ubuntu 16.04)
# This script will modify the configuration of the system and install things
# Should be run as root
#==============================================================================

set -e

function main() {
    prepare_machine "$@"
    prepare_docker_image # Correspond docker-pytorch-linux-bionic-py3.6-clang9 CircleCI job
    build # Correspond pytorch_linux_bionic_py3_6_clang9_build CircleCI job
    echo -e "===========================\n  BUILD SCRIPT SUCCEEDED\n===========================\n"
    poweroff
}

# Install dependencies and create circleci user
function prepare_machine() {
    # Installing deps:
    # docker, jq to parse aws secretsmanager and moreutils and expect-dev for ts and unbuffer
    apt-get install -y docker.io jq moreutils expect-dev

    # Install pip command as pip3
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
}

# Build Docker image
# See: https://app.circleci.com/pipelines/github/pytorch/pytorch/201040/workflows/05547b7e-f7c7-447e-b6cf-0158d15bc6e3/jobs/6745572
function prepare_docker_image() {
    mkdir -p /home/$USER/project
    cd /home/$USER/project
    git clone https://github.com/pytorch/pytorch .
    git checkout 490d41aaa61a9c0b12637e40cec066bf0e9515f3 # patchs regularly get broken

    # Add verificarlo to docker image
    patch_docker_build_scripts

    docker build \
        --no-cache \
        --progress=plain \
        --build-arg TRAVIS_DL_URL_PREFIX=https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64 \
        --build-arg BUILD_ENVIRONMENT=pytorch-linux-bionic-py3.6-verificarlo \
        --build-arg PROTOBUF=yes \
        --build-arg THRIFT= \
        --build-arg LLVMDEV=yes \
        --build-arg DB=yes \
        --build-arg VISION=yes \
        --build-arg EC2= \
        --build-arg JENKINS= \
        --build-arg JENKINS_UID= \
        --build-arg JENKINS_GID= \
        --build-arg UBUNTU_VERSION=18.04 \
        --build-arg CENTOS_VERSION= \
        --build-arg DEVTOOLSET_VERSION= \
        --build-arg GLIBC_VERSION= \
        --build-arg CLANG_VERSION=9 \
        --build-arg VERIFICARLO_VERSION=github \
        --build-arg ANACONDA_PYTHON_VERSION=3.6 \
        --build-arg TRAVIS_PYTHON_VERSION= \
        --build-arg GCC_VERSION= \
        --build-arg CUDA_VERSION= \
        --build-arg CUDNN_VERSION= \
        --build-arg ANDROID= \
        --build-arg ANDROID_NDK= \
        --build-arg GRADLE_VERSION= \
        --build-arg VULKAN_SDK_VERSION= \
        --build-arg SWIFTSHADER= \
        --build-arg CMAKE_VERSION= \
        --build-arg NINJA_VERSION= \
        --build-arg KATEX= \
        --build-arg ROCM_VERSION= \
        -f .circleci/docker/ubuntu/Dockerfile \
        -t fuzzy-pytorch-buildenv \
        .circleci/docker
}

function patch_docker_build_scripts() {
cat << 'EOF' | git apply
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
+  git checkout 09b24e04797dcf849ca1080d8d06e6d89a14dc65
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

function build() {
    export CI=true \
           CIRCLECI=true

    # Reusing source code cloned in previous "CircleCI Job"

    # See pytorch_params section of CircleCI config
    export BUILD_ENVIRONMENT="pytorch-linux-bionic-py3.6-verificarlo-build"
    export DOCKER_IMAGE="fuzzy-pytorch-buildenv"
    export USE_CUDA_RUNTIME=""
    export BUILD_ONLY=""

    # Attempting to replace the script setup_ci_environment.sh by:
    cat << EOF > /home/$USER/project/env
IN_CIRCLECI=1
BUILD_ENVIRONMENT=pytorch-linux-bionic-py3.6-verificarlo-build
MAX_JOBS=$(($(nproc) - 1))
EOF


    echo "Launching the build docker container (${DOCKER_IMAGE}):"
    export id=$(docker run -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})
    git submodule sync && git submodule update -q --init --recursive

    # Customizations:
    patch_build_script_to_handle_verificarlo
    setup_function_instrumentation $id
    disable_blas

    docker cp /home/$USER/project/. $id:/var/lib/jenkins/workspace

    export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "set -a && source ./workspace/env && set +a" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/build.sh && find ${BUILD_ROOT} -type f -name "*.a" -or -name "*.o" -or -name "*.ll" -delete") | docker exec -u jenkins -i "$id" bash) 2>&1'

    echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

    # Commit built Docker image
    COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-built
    docker commit "$id" ${COMMIT_DOCKER_IMAGE}
    push_final_docker_image ${COMMIT_DOCKER_IMAGE}
}

function push_final_docker_image() {
	build_docker_image=$1
	cd ~
	mkdir -p fuzzy-pytorch
	cat << 'EOF' > fuzzy-pytorch/Dockerfile
FROM ubuntu:18.04

RUN     apt-get update && \
        apt-get install -y wget && \
        wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
        chmod +x ./Miniconda3-latest-Linux-x86_64.sh && \
        ./Miniconda3-latest-Linux-x86_64.sh -b -f -p "/opt/conda" && \
        /opt/conda/bin/conda install -y python="3.6" numpy=1.18.5 pyyaml mkl mkl-include setuptools \
                                        cffi typing typing_extensions future six dataclasses && \
        rm -rf /var/lib/apt/lists/* Miniconda3-latest-Linux-x86_64.sh

ENV     PATH="/opt/conda/bin:${PATH}"

COPY    --from=fuzzy-pytorch-builder \
        /opt/conda/lib/python3.6/site-packages/torch /opt/conda/lib/python3.6/site-packages/torch
COPY    --from=fuzzy-pytorch-builder \
        /usr/local/lib/libinterflop_*.so /usr/local/lib/
COPY    --from=fuzzy-pytorch-builder \
        /usr/lib/x86_64-linux-gnu/libomp.so.5 /usr/lib/x86_64-linux-gnu/

ENV     VFC_BACKENDS="libinterflop_mca.so"

ENTRYPOINT bash
EOF
	docker tag $build_docker_image fuzzy-pytorch-builder
	docker build fuzzy-pytorch -t fuzzy-pytorch
	test_fuzzy_pytorch "fuzzy-pytorch"
	if [ "$?" -eq "0" ] ; then # Tests passed
		DOCKERHUB_TOKEN=$(aws secretsmanager get-secret-value --region us-east-2 --secret-id DockerToken | jq -r '.SecretString')
		echo $DOCKERHUB_TOKEN | docker login --username hantoine --password-stdin
		docker tag fuzzy-pytorch hantoine/fuzzy-pytorch
		docker push hantoine/fuzzy-pytorch
		push_image_with_jupyterlab
	fi
}

push_image_with_jupyterlab() {
	mkdir -p fuzzy-pytorch-jupyter
	cat << 'EOF' > fuzzy-pytorch-jupyter/Dockerfile
FROM hantoine/fuzzy-pytorch

RUN conda install -y jupyterlab
EXPOSE 8888/tcp

ENTRYPOINT ["/opt/conda/bin/jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
EOF
	docker build fuzzy-pytorch-jupyter -t hantoine/fuzzy-pytorch:jupyter
	docker push hantoine/fuzzy-pytorch:jupyter
}

function test_fuzzy_pytorch() {
docker_image=$1
cat << 'EOF' | docker run -i $docker_image bash
VFC_BACKENDS="libinterflop_mca.so" python -c "
import torch
import pickle

N_SAMPLES = 30

torch.manual_seed(0)
a = torch.rand(5, 5)
b = torch.rand(5, 5)

res = []
for _ in range(N_SAMPLES):
  res.append(a @ b)

with open('fuzzy-pytorch_mca_results.pickle', 'wb') as file:
  pickle.dump(torch.stack(res), file)
"
VFC_BACKENDS="libinterflop_ieee.so" python -c "
import torch
import pickle

torch.manual_seed(0)
a = torch.rand(5, 5)
b = torch.rand(5, 5)
correct_res = a @ b

with open('fuzzy-pytorch_mca_results.pickle', 'rb') as file:
  mca_res = pickle.load(file)

mean_res = mca_res.mean(dim=0)
relative_errors = (mean_res - correct_res) / correct_res
print('Relative errors: ')
print(relative_errors)

assert torch.allclose(mean_res, correct_res), 'Results of matrix multiplication with MCA not centered on the correct result'
print('[PASSED] Results of matrix multiplication with MCA centered on the correct result')
assert mca_res.std(dim=0).sum() != 0, 'Results of matrix multiplication with MCA are deterministic'
print('[PASSED] Results of matrix multiplication with MCA are not deterministic')
"
EOF
return $?
}

main "$@"
