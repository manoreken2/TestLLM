# GPT2 model Training experiment with Japanese text

This program is derived from the book [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D).

Original book sourcecode: https://github.com/rasbt/LLMs-from-scratch

## Computer setup

Tested on x64 PC, Ubuntu Linux 24.04.3, NVIDIA RTX 5090

Install MiniForge https://github.com/conda-forge/miniforge

Create conda environment for gpt2_train_jp

```bash
conda create -n gpt2trainjp python=3.12
conda activate gpt2trainjp
```

Install PyTorch with Cuda enabled https://pytorch.org/

```bash
conda activate gpt2trainjp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

## Prepare dataset

Converts Aozora bunko text to UTF-8

```bash
conda activate gpt2trainjp
python 00_cleanup.py
```

Put UTF-8 encoded txt onto orig directory then run

```bash
conda activate gpt2trainjp
python 01_prepare_dataset.py
```


## Train

Set configuration params onto train_conf.yaml

Then run

```bash
python 02_train.py
```

