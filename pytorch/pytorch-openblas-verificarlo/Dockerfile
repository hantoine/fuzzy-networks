FROM openblas:verificarlo

ENV PYTHON_VERSION 3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# PYTHONPATHH was previously set for verificarlo build and could confuse conda
ENV PYTHONPATH= 
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION nomkl conda-build pyyaml numpy ipython && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


ENV BLAS=OpenBLAS \
    USE_CUDA=OFF \
    USE_NUMA=OFF \
    USE_MPI=OFF \
    USE_NCCL=OFF \
    USE_METAL=OFF \
    USE_FAKELOWP=OFF \
    USE_FBGEMM=OFF


WORKDIR /opt
# Patch fixing non-relevant warning message about disabling fakelowp when already disabled
COPY Dependencies.cmake.patch .
# Environement variables CC, CXX, LD and LDSHARED are unset to stop using verificarlo to compile Pytorch
RUN git clone https://github.com/pytorch/pytorch && \
    cd pytorch && git submodule sync && git submodule update --init --recursive && \
    cd cmake && patch < ../../Dependencies.cmake.patch && cd .. && \
    unset CC CXX LD LDSHARED && \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install && \
    pip install -v . && \
    rm -rf /opt/pytorch /opt/Dependencies.cmake.patch

WORKDIR /workdir
