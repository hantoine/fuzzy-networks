FROM hantoine/fuzzy-pytorch

RUN conda install absl-py

WORKDIR /workdir
ADD . .
VOLUME /mca_predictions

ENTRYPOINT ["./run_mca_trials_parallel.py"]
