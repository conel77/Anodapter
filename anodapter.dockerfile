
FROM nvcr.io/nvidia/pytorch:21.07-py3


ENV DEBIAN_FRONTEND=noninteractive


RUN apt update && \
    apt install -y git && \
    pip install --no-cache-dir \
        git+https://github.com/huggingface/diffusers.git@6a5f06488c0d88c1827c016835cd5f64abe4b52c \
        transformers==4.38.2 \
        opencv-python==4.5.1.48 \
        scikit-image==0.21.0 \
        pytorch-lightning==1.5.9 \
        timm==1.0.7 \
        xformers==0.0.26.post1 \
        einops==0.4.1 \
        imageio \
        imageio-ffmpeg \
        open-clip-torch==2.24.0 \
        lpips==0.1.4 \
        bitsandbytes==0.43.1 \
        torchmetrics==0.6.0 \
        peft==0.8.0 \
        accelerate==0.31.0 \
        datasets==2.20.0 \
        huggingface-hub==0.23.3 \
        safetensors==0.4.5
