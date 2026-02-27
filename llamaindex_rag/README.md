## Setup

install Minforge https://github.com/conda-forge/miniforge


```
conda create -y -n Lidx python=3.11
conda activate Lidx
pip install llama-index llama-index-llms-huggingface llama-index-readers-web llama-index-embeddings-huggingface notebook ipywidgets widgetsnbextension pandas-profiling hf_xet
```

## Run

Set environment variable CUDA_VISIBLE_DEVICES
```
# Ubuntu 24.04
export CUDA_VISIBLE_DEVICES=""

# Windows
set CUDA_VISIBLE_DEVICES=""
```
Then run jupyter notebook

```
jupyter notebook Run.ipynb
```
