## Setup

Tested on Windows PC

install Minforge https://github.com/conda-forge/miniforge


```
conda create -y -n Lidx python=3.11
conda activate Lidx
pip install triton-windows==3.6.0.post26 llama-index llama-index-llms-huggingface llama-index-readers-web llama-index-embeddings-huggingface notebook ipywidgets widgetsnbextension pandas-profiling hf_xet qdrant_client transformers llama-index-llms-ollama torch==2.10.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

```

## Run

When GPU out of memory happens, set environment variable CUDA_VISIBLE_DEVICES to run on CPU.

```
# Ubuntu 24.04
export CUDA_VISIBLE_DEVICES=""

# Windows
set CUDA_VISIBLE_DEVICES=""
```

Run jupyter notebook

```
cd /d C:\work\TestLLM\llamaindex_rag
conda activate Lidx
jupyter notebook Run_gptoss_Saiyuuki.ipynb

```

Remember to delete CUDA_VISIBLE_DEVICES, otherwise your computer cannot use CUDA.

