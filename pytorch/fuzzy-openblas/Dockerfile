FROM fuzzy:base-latest

RUN apt-get update &&\
    apt-get install -y build-essential &&\
    rm -rf /var/lib/apt/lists/ &&\
    mkdir -p /opt/build

# ENV CC "verificarlo"
# ENV CXX "verificarlo"
# ENV FC "verificarlo"
# ENV LD "verificarlo"
# ENV LDSHARED "verificarlo -shared"
# ENV HOSTCC "verificarlo"

RUN cd /opt/build/ &&\
    wget --quiet https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz &&\
    tar xf v0.3.9.tar.gz &&\
    cd /opt/build/OpenBLAS-0.3.9/ &&\
    make &&\
    make install PREFIX=/usr/local &&\
    rm -rf /opt/build &&\
    echo "/opt/OpenBLAS/lib" >> /etc/ld.so.conf.d/openblas.conf &&\
    ldconfig 
