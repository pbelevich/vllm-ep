# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

################################ NCCL ########################################

ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.45.0
ARG NCCL_VERSION=v2.28.9-1
ARG NCCL_TESTS_VERSION=v2.17.6

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev

RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

ENV OPAL_PREFIX=

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    check \
    cmake \
    curl \
    debhelper \
    devscripts \
    git \
    gcc \
    gdb \
    kmod \
    libsubunit-dev \
    libtool \
    openssh-client \
    openssh-server \
    pkg-config \
    python3-distutils \
    vim \
    python3.10-dev \
    python3.10-venv
RUN apt-get purge -y cuda-compat-*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && pip3 install awscli pynvml

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include:$CPATH
ENV PATH=/opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install NCCL
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git  /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

###################################################
## Install NCCL-tests
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) \
    MPI=1 \
    MPI_HOME=/opt/amazon/openmpi/ \
    CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/opt/nccl/build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

RUN rm -rf /var/lib/apt/lists/*

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

## Set LD_PRELOAD for NCCL library
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so

################################ Miniconda ########################################

# Install Miniconda to not depend on the base image python
RUN mkdir -p /opt/miniconda3 \
    && curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/miniconda3 \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && /opt/miniconda3/bin/conda init bash

ENV PATH="/opt/miniconda3/bin:${PATH}"

RUN python --version

################################ NVSHMEM ########################################

ENV NVSHMEM_DIR=/opt/nvshmem
ENV NVSHMEM_HOME=/opt/nvshmem

# 3.2.5-1: wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz && tar -xvf nvshmem_src_3.2.5-1.txz
# 3.3.9:   wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.3.9/source/nvshmem_src_cuda12-all-all-3.3.9.tar.gz && tar -xvf nvshmem_src_cuda12-all-all-3.3.9.tar.gz
# 3.4.5-0: git clone https://github.com/NVIDIA/nvshmem.git && cd ./nvshmem && git checkout v3.4.5-0

ARG NVSHMEM_REPO_URL=https://github.com/NVIDIA/nvshmem.git
ARG NVSHMEM_COMMIT=v3.4.5-0

RUN git clone ${NVSHMEM_REPO_URL} /nvshmem_src \
    && cd /nvshmem_src \
    && git checkout ${NVSHMEM_COMMIT} \
    && mkdir -p build \
    && cd build \ 
    && cmake \
    -DNVSHMEM_PREFIX=/opt/nvshmem \
    -DCMAKE_INSTALL_PREFIX=/opt/nvshmem \
    \
    -DCUDA_HOME=/usr/local/cuda \
    -DCMAKE_CUDA_ARCHITECTURES="90a;100a" \
    \
    -DNVSHMEM_USE_GDRCOPY=1 \
    -DGDRCOPY_HOME=/opt/gdrcopy \
    \
    -DNVSHMEM_USE_NCCL=1 \
    -DNCCL_HOME=/opt/nccl/build \
    -DNCCL_INCLUDE=/opt/nccl/build/include \
    \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DMPI_HOME=/opt/amazon/openmpi \
    \
    -DNVSHMEM_PMIX_SUPPORT=1 \
    -DPMIX_HOME=/opt/amazon/pmix \
    -DNVSHMEM_DEFAULT_PMIX=1 \
    \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=1 \
    \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    \
    -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    .. \
    && make -j$(nproc) \
    && make install

ENV PATH=/opt/nvshmem/bin:$PATH LD_LIBRARY_PATH=/opt/amazon/pmix/lib:/opt/nvshmem/lib:$LD_LIBRARY_PATH NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa

## Set LD_PRELOAD for NVSHMEM library
ENV LD_PRELOAD=/opt/nvshmem/lib/libnvshmem_host.so:$LD_PRELOAD

################################ extra packages ########################################

RUN pip install ninja numpy cmake pytest blobfile datasets

################################ Rust ########################################

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.91.0 --component llvm-tools-preview
ENV PATH="/root/.cargo/bin:$PATH" \
    CARGO_HOME="/root/.cargo" \
    RUSTUP_HOME="/root/.rustup"

################################ PyTorch ########################################

ARG TORCH_VERSION=2.9.1

RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu128

################################ flashInfer ########################################

RUN pip install flashinfer-python

################################ DeepGEMM ########################################

# see: https://github.com/deepseek-ai/DeepGEMM#installation

ARG DEEPGEMM_REPO_URL=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_COMMIT=38f8ef73a48a42b1a04e0fa839c2341540de26a6

RUN git clone ${DEEPGEMM_REPO_URL} /DeepGEMM \
    && cd /DeepGEMM \
    && git checkout ${DEEPGEMM_COMMIT} \
    && git submodule update --init --recursive \
    && ./install.sh

################################ vLLM ########################################

ARG VLLM_REPO_URL=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT=main

RUN git clone ${VLLM_REPO_URL} /vllm \
    && cd /vllm \
    && git checkout ${VLLM_COMMIT} \
    && python use_existing_torch.py \
    && pip install -r requirements/build.txt \
    && pip install --no-build-isolation -v -e .

################################ PPLX KERNELS ########################################

# see: https://github.com/vllm-project/vllm/tree/main/tools/ep_kernels
# see: https://github.com/pbelevich/pplx-kernels-benchmark

ARG PPLX_KERNELS_REPO_URL=https://github.com/ppl-ai/pplx-kernels.git
ARG PPLX_KERNELS_COMMIT=12cecfda252e4e646417ac263d96e994d476ee5d

RUN git clone ${PPLX_KERNELS_REPO_URL} /pplx-kernels \
    && cd /pplx-kernels \
    && git checkout ${PPLX_KERNELS_COMMIT}

RUN cd /pplx-kernels \
    && TORCH_CUDA_ARCH_LIST="9.0a+PTX;10.0a+PTX" python3 setup.py bdist_wheel \
    && pip install dist/*.whl

################################ PPLX Garden ########################################

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libclang-dev \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python3-build \
    python3-venv \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV GDRAPI_HOME=/opt/gdrcopy
ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH

ARG PPLX_GARDEN_REPO_URL=https://github.com/perplexityai/pplx-garden.git
ARG PPLX_GARDEN_COMMIT=f62aac558ef937340d17091a52011631d0c65147

RUN git clone ${PPLX_GARDEN_REPO_URL} /pplx-garden \
    && cd /pplx-garden \
    && git checkout ${PPLX_GARDEN_COMMIT} \
    && export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
    && python3 -m build --wheel \
    && python3 -m pip install /pplx-garden/dist/*.whl

################################ DeepEP ########################################

ARG DEEPEP_REPO_URL=https://github.com/pbelevich/DeepEP.git
ARG DEEPEP_COMMIT=27e8e661857499068275dbaa09e4c15d67d51f81

RUN git clone ${DEEPEP_REPO_URL} /DeepEP \
    && cd /DeepEP \
    && git checkout ${DEEPEP_COMMIT} \
    && TORCH_CUDA_ARCH_LIST="9.0a+PTX;10.0a+PTX" python3 setup.py install --prefix=/DeepEP/install

################################ UCCL-EP ########################################

RUN apt-get update && apt-get install -y \
    sudo \
    libnuma-dev

ARG UCCL_REPO_URL=https://github.com/uccl-project/uccl.git
ARG UCCL_COMMIT=e4d2b2e00aed5c7b6b7d5f908a579e49ea538048

RUN ls -la && \
    git clone ${UCCL_REPO_URL} /uccl \
    && cd /uccl \
    && git checkout ${UCCL_COMMIT} \
    && cd ep \
    && apt install -y nvtop libgoogle-glog-dev clang-format-14 python3-pip \
    && TORCH_CUDA_ARCH_LIST="9.0a+PTX;10.0a+PTX" python setup.py install --prefix=/uccl/install \
    && cd deep_ep_wrapper \
    && TORCH_CUDA_ARCH_LIST="9.0a+PTX;10.0a+PTX" python setup.py install --prefix=/uccl/install
