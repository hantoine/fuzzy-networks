#!/bin/bash

# Build Fuzzy-PyTorch docker image
docker build --build-arg MAX_JOBS=$(($(nproc) - 1)) -t fuzzy-pytorch-totest docker

# Test the image
cat test_fuzzy_pytorch.sh | docker run -i fuzzy-pytorch-totest
if [ "$?" -ne "0" ] ; then
    print "Tests have failed"
    exit 1
fi
docker tag fuzzy-pytorch-totest fuzzy-pytorch
