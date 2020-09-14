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
    # Installing docker
    apt-get install -y docker.io

    # Logging in to Docker Hub
    aws secretsmanager get-secret-value --region us-east-2 --secret-id DockerToken --output text \
        | head -n 1 | cut  -f 4 \
        | docker login --username hantoine --password-stdin
}

# Build Docker image
# See: https://app.circleci.com/pipelines/github/pytorch/pytorch/201040/workflows/05547b7e-f7c7-447e-b6cf-0158d15bc6e3/jobs/6745572
function prepare_docker_image() {
    git clone https://github.com/hantoine/fuzzy-networks
    cd fuzzy-networks
    buildenv_path="pytorch/build/buildenv"
    docker_tag=$(git rev-parse HEAD:$buildenv_path)

    set +e
    docker pull hantoine/fuzzy-pytorch-buildenv:$docker_tag
    docker_pull_ret=$?
    set -e
    if [ "$docker_pull_ret" -ne "0" ] ; then

        docker build -t fuzzy-pytorch-buildenv $buildenv_path
        docker tag fuzzy-pytorch-buildenv hantoine/fuzzy-pytorch-buildenv:$docker_tag
        docker push hantoine/fuzzy-pytorch-buildenv:$docker_tag
    else
        docker tag hantoine/fuzzy-pytorch-buildenv:$docker_tag fuzzy-pytorch-buildenv
    fi
    cd ..
    echo "====================================="
    echo "  Docker image buildenv built"
    echo "====================================="
}

function extract_build_script_patch() {
cat << 'EOF' > build_script_patch
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

function extract_script_setting_up_function_instrumentation() {
cat << 'EOFF' > script_setting_up_function_instrumentation
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

function extract_patch_disabling_blas() {

cat << 'EOF' > patch_disabling_blas
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
    extract_build_script_patch
    extract_patch_disabling_blas
    extract_script_setting_up_function_instrumentation

    cat << EOF > Dockerfile
FROM fuzzy-pytorch-buildenv

# WORKDIR would create it but as root
RUN mkdir /var/lib/jenkins/workspace

WORKDIR /var/lib/jenkins/workspace
RUN git clone https://github.com/pytorch/pytorch . && \
    git checkout 490d41aaa61a9c0b12637e40cec066bf0e9515f3 && \
    git submodule sync && git submodule update -q --init --recursive

# Patch build script
ADD build_script_patch .
RUN git apply build_script_patch

# Disable BLAS
ADD patch_disabling_blas .
RUN git apply patch_disabling_blas

# Setup function instrumentation
ADD script_setting_up_function_instrumentation .
RUN bash script_setting_up_function_instrumentation

ENV IN_CIRCLECI=1 \
    BUILD_ENVIRONMENT=pytorch-linux-bionic-py3.6-verificarlo-build \
    MAX_JOBS=$(($(nproc) - 1))
RUN .jenkins/pytorch/build.sh && \
    find . -type f -name "*.a" -or -name "*.o" -or -name "*.ll" -delete

EOF
    docker build . -t fuzzy-pytorch-built
    push_final_docker_image
}

function push_final_docker_image() {
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

COPY    --from=fuzzy-pytorch-built \
        /opt/conda/lib/python3.6/site-packages/torch /opt/conda/lib/python3.6/site-packages/torch
COPY    --from=fuzzy-pytorch-built \
        /usr/local/lib/libinterflop_*.so /usr/local/lib/
COPY    --from=fuzzy-pytorch-built \
        /usr/lib/x86_64-linux-gnu/libomp.so.5 /usr/lib/x86_64-linux-gnu/

ENV     VFC_BACKENDS="libinterflop_mca.so"

ENTRYPOINT bash
EOF
	docker build fuzzy-pytorch -t fuzzy-pytorch
	test_fuzzy_pytorch "fuzzy-pytorch"
	if [ "$?" -eq "0" ] ; then # Tests passed
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
