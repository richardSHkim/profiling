FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

# Install MMDetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .
# copy init file for version matching
COPY init_version_fixed.py /mmdetection/mmdet/__init__.py

# Install fvcore
RUN pip install fvcore==0.1.5.post20221221

# Install Calflops
RUN pip install calflops==0.3.2 transformers

# Install DeepSpeed
RUN pip install deepspeed==0.16.7

# git clone
COPY . /profiling
# RUN git clone https://github.com/richardSHkim/profiling /profiling
WORKDIR /profiling

ENTRYPOINT ["python", "check_profilers.py"]