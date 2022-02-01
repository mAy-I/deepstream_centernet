## Prerequisites

- Ubuntu 18.04 or later
- Docker 19.03 or later
- Docker image : [nvcr.io/nvidia/pytorch:21.10-py3](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
- Nvidia Graphic Driver 470+

## Installation

### Prepare environment

1. Run [nvcr.io/nvidia/pytorch:21.10-py3](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
    ```shell
    docker run --gpus all --ipc=host --rm -it nvcr.io/nvidia/pytorch:21.10-py3
    ```

2. Activate conda virtual environment
    ```shell
    source /opt/conda/bin/activate
    ```

### Install

1. Install MMCV(modified it to use only DCN and update tensorrt 8)
    ```shell
    export TENSORRT_DIR=/usr/local/cuda
    git clone https://github.com/jaemin93/mmcv.git
    cd mmcv && MMCV_WITH_OPS=1 MMCV_WITH_TRT=1 pip install -e .
    ```
    Note: If you don't use a DCNv2 module, skip it.

2. Install Gstreamer
    ```shell
    export DEBIAN_FRONTEND=noninteractive
    apt-get install python3-gi \
        python3-gst-1.0 \
        libgirepository1.0-dev \
        libcairo2-dev \
        gir1.2-gstreamer-1.0 \
        libssl1.1 \
        libgstreamer1.0-0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstrtspserver-1.0-0 \
        libjansson4 \
        python3-dev \
        python-gi-dev \
        python-dev \
        libglib2.0-dev \
        libglib2.0-dev-bin \
        libtool m4 autoconf automake
    ```

3. Install Deepstream SDK 6.0

    Download [.deb file](https://developer.nvidia.com/deepstream-sdk) in DeepStream 6.0 for Servers and Workstations
    ```shell
    apt-get install ./deepstream-6.0_6.0.0-1_amd64.deb
    ```

4. Install pyds
    ```shell
    pip install wheel/pyds-1.1.0-py3-none-linux_x86_64.whl
    ```
    Note : Please refer to [NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md)



