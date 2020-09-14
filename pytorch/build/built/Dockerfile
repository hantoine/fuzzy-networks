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
    BUILD_ENVIRONMENT=pytorch-linux-bionic-py3.6-verificarlo-build

ARG MAX_JOBS

RUN .jenkins/pytorch/build.sh && \
    find . -type f -name "*.a" -or -name "*.o" -or -name "*.ll" -delete

CMD ['bash']

