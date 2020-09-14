ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION=18.04

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
ADD ./install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV=yes
ARG CLANG_VERSION=9
ADD ./install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install user
ADD ./install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda
ENV PATH /opt/conda/bin:$PATH
ARG ANACONDA_PYTHON_VERSION=3.6
ADD ./install_conda.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Install Verificarlo
ARG VERIFICARLO_VERSION=github
ADD ./install_verificarlo.sh install_verificarlo.sh
RUN bash ./install_verificarlo.sh && rm install_verificarlo.sh

# (optional) Install protobuf for ONNX
ARG PROTOBUF=yes
ADD ./install_protobuf.sh install_protobuf.sh
RUN if [ -n "${PROTOBUF}" ]; then bash ./install_protobuf.sh; fi
RUN rm install_protobuf.sh
ENV INSTALLED_PROTOBUF ${PROTOBUF}

# (optional) Install database packages like LMDB and LevelDB
ARG DB=yes
ADD ./install_db.sh install_db.sh
RUN if [ -n "${DB}" ]; then bash ./install_db.sh; fi
RUN rm install_db.sh
ENV INSTALLED_DB ${DB}

# (optional) Install vision packages like OpenCV and ffmpeg
ARG VISION=yes
ADD ./install_vision.sh install_vision.sh
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh
ENV INSTALLED_VISION ${VISION}

# Install ccache/sccache (do this last, so we get priority in PATH)
ADD ./install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

# Install LLVM dev version (Defined in the pytorch/builder github repository)
COPY --from=pytorch/llvm:9.0.1 /opt/llvm /opt/llvm

USER jenkins
CMD ["bash"]
