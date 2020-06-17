FROM verificarlo:hantoine-openblas

RUN apt-get update &&\
    apt-get install -y build-essential wget &&\
    rm -rf /var/lib/apt/lists/ &&\
    mkdir -p /opt/build

ENV CC="verificarlo" \
    CXX="verificarlo" \
    FC="verificarlo" \
    LD="verificarlo" \
    LDSHARED="verificarlo -shared" \
    HOSTCC="gcc"

RUN cd /opt/build/ &&\
    wget --quiet https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz &&\
    tar xf v0.3.9.tar.gz &&\
    cd /opt/build/OpenBLAS-0.3.9/ &&\
    make &&\
    make install PREFIX=/usr/local &&\
    cd / && rm -rf /opt/build &&\
    echo "/opt/OpenBLAS/lib" >> /etc/ld.so.conf.d/openblas.conf &&\
    ldconfig 
