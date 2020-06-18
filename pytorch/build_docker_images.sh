#!/bin/sh
docker build -t openblas-mca openblas-verificarlo
docker tag openblas-mca hantoine/openblas-mca
docker build -t pytorch-mca pytorch-openblas-verificarlo
docker tag pytorch-mca hantoine/pytorch-mca
