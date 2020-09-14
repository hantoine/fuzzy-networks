FROM hantoine/fuzzy-pytorch

RUN conda install -y jupyterlab
EXPOSE 8888/tcp

ENTRYPOINT ["/opt/conda/bin/jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
