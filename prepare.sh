
#!/usr/bin/env bash
set -x

export PS1=""
conda create -n verl python=3.10 -y
conda activate verl

USE_MEGATRON=0

USE_SGLANG=1

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    pip install flashinfer_python==0.2.3
    pip install "sglang[all]==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && pip install torch-memory-saver --no-cache-dir
fi
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
fi


echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overrided)"
    pip install nvidia-cudnn-cu12==9.8.0.87
fi

echo "Successfully installed all packages"


pip install --no-deps -e .

#prepare data
# python recipe/ours/data_preprocess/deepscaler/deepscaler_dataset.py --local_dir='./data/deepscaler'
# python recipe/ours/data_preprocess/countdown.py --local_dir='./data/countdown3to4'
# python recipe/ours/data_preprocess/countdown4.py --local_dir='./data/countdown4'
# python recipe/ours/data_preprocess/geo3k.py --local_dir='./data/geo3k'
# python recipe/ours/data_preprocess/math_dataset.py --local_dir='./data/math'


# download model
pip install -U huggingface_hub
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B
# huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir models/DeepSeek-R1-Distill-Qwen-7B
# huggingface-cli download --resume-download Qwen/Qwen2.5-3B --local-dir models/Qwen2.5-3B
# huggingface-cli download --resume-download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/Qwen2.5-VL-3B-Instruct


pip install tabulate

# wandb
pip install wandb
