FROM hantoine/verificarlo

ENV OPENBLAS_VERSION 0.3.10

RUN apt-get update &&\
    apt-get install -y build-essential wget &&\
    rm -rf /var/lib/apt/lists/ &&\
    mkdir -p /opt/build

ENV CC="verificarlo-c" \
    CXX="verificarlo-c++" \
    FC="verificarlo-f" \
    HOSTCC="gcc" \
    VFC_BACKENDS="libinterflop_ieee.so" \
    MAKE_NB_JOBS="1"

COPY f_check.patch /opt/build/

RUN cd /opt/build/ &&\
    wget --quiet https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz &&\
    tar xf v${OPENBLAS_VERSION}.tar.gz &&\
    cd /opt/build/OpenBLAS-${OPENBLAS_VERSION}/ &&\
    patch < /opt/build/f_check.patch &&\
    make &&\
    make install PREFIX=/usr/local &&\
    cd / && rm -rf /opt/build &&\
    echo "/opt/OpenBLAS/lib" >> /etc/ld.so.conf.d/openblas.conf &&\
    ldconfig 

ENV VFC_BACKENDS="libinterflop_mca.so"
