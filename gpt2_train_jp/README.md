GPT2 model Training experiment with Japanese text

program is derived from the book [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D).
original book sourcecode: https://github.com/rasbt/LLMs-from-scratch

## Computer setup

Install MiniForge https://github.com/conda-forge/miniforge

Create conda environment for gpt2_train_jp

conda create -n gpt2trainjp python=3.12
conda activate gpt2trainjp

Install PyTorch with Cuda enabled https://pytorch.org/

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

## Prepare dataset

Converts Aozora bunko text to UTF-8

python 00_cleanup.py

Put UTF-8 encoded txt onto orig directory then run

python 01_prepare_dataset.py



## Train

Set configuration params onto train_conf.yaml

Then run

python 02_train.py


