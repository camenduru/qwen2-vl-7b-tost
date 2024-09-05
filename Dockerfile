FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    git+https://github.com/huggingface/transformers transformers-stream-generator==0.0.5 accelerate==0.33.0 bitsandbytes==0.43.3 sentencepiece==0.2.0 protobuf==5.28.0 qwen-vl-utils==0.0.2 runpod && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/chat_template.json -d /content/Qwen2-VL-7B-Instruct -o chat_template.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/config.json -d /content/Qwen2-VL-7B-Instruct -o config.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/generation_config.json -d /content/Qwen2-VL-7B-Instruct -o generation_config.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/merges.txt -d /content/Qwen2-VL-7B-Instruct -o merges.txt  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/resolve/main/model-00001-of-00005.safetensors -d /content/Qwen2-VL-7B-Instruct -o model-00001-of-00005.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/resolve/main/model-00002-of-00005.safetensors -d /content/Qwen2-VL-7B-Instruct -o model-00002-of-00005.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/resolve/main/model-00003-of-00005.safetensors -d /content/Qwen2-VL-7B-Instruct -o model-00003-of-00005.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/resolve/main/model-00004-of-00005.safetensors -d /content/Qwen2-VL-7B-Instruct -o model-00004-of-00005.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/resolve/main/model-00005-of-00005.safetensors -d /content/Qwen2-VL-7B-Instruct -o model-00005-of-00005.safetensors  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/model.safetensors.index.json -d /content/Qwen2-VL-7B-Instruct -o model.safetensors.index.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/preprocessor_config.json -d /content/Qwen2-VL-7B-Instruct -o preprocessor_config.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/tokenizer.json -d /content/Qwen2-VL-7B-Instruct -o tokenizer.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/tokenizer_config.json -d /content/Qwen2-VL-7B-Instruct -o tokenizer_config.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Qwen2-VL-7B-Instruct/raw/main/vocab.json -d /content/Qwen2-VL-7B-Instruct -o vocab.json

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py
