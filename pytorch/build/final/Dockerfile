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

CMD ["bash"]
