#!/bin/sh
docker build docker -t mca_mlp_numpy
docker run -ti --mount type=bind,src=$(readlink -f mca_preds),dst=/mca_predictions mca_mlp_numpy
