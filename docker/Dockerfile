ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del A4B469963BF863CC
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get install -y aria2 libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6
#RUN apt-get update && apt-get install -y ffmpeg

# Install MMDetection
RUN git clone --branch v1.0.0rc2_transfusion https://github.com/mljack/TransFusion.git /TransFusion
WORKDIR /TransFusion

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
#ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN python -m pip install --upgrade pip
RUN pip install -r requirements/mminstall.txt
RUN pip install -r requirements/build.txt
#RUN pip install -r requirements/optional.txt
RUN pip install -r requirements/runtime.txt
RUN pip install -r requirements/tests.txt
#RUN pip install open3d
pip install opencv-python-headless

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip install cython
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force

RUN pip install --no-cache-dir -e .
