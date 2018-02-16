FROM continuumio/miniconda3:4.3.14
MAINTAINER Ziga Avsec <avsec@in.tum.de>

RUN apt-get update && \
    apt-get install -y build-essential libz-dev libcurl3-dev

# install git-lfs
RUN echo "deb http://http.debian.net/debian wheezy-backports main" > /etc/apt/sources.list.d/wheezy-backports-main.list && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# install conda deps5
RUN conda install --yes numpy pandas pytest h5py scipy scikit-learn && \
    conda install --yes -c bioconda pybigwig pybedtools pyvcf cyvcf2 bedtools htslib && \
    conda install --yes pytorch-cpu -c pytorch && \
    conda install --yes -c conda-forge keras tensorflow && \
    conda clean --yes -all

RUN pip install cython bcolz>=1.1 tqdm concise && \
    pip install matplotlib -U && \
    rm -rf /root/.cache/pip/*
