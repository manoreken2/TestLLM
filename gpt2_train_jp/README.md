# GPT2 model Training experiment with Japanese text

This program is derived from the book [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D).

Original book sourcecode: https://github.com/rasbt/LLMs-from-scratch

## Computer setup

Tested on x64 PC, Ubuntu Linux 24.04.3, NVIDIA RTX 5090

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y nvidia-driver-580-open
sudo reboot
```

On the next boot, check the gpu is recognized correctly

```bash
nvidia-smi
```

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

Download text files from Aozora bunko https://www.aozora.gr.jp/

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

## Predict

Run 03_predict.py to predict text

```bash
python 03_predict.py --input_str "僕は"

PyTorch version 2.8.0+cu129. Using cuda device. CUDA version: 12.9. high matmul precision. 
    僕は人並みにリュック・サックを背負い、あの上高地の温泉宿から穂高山へ登ろうとしました。穂高山へ登るのには御承知のとおり梓川をさかのぼるほかはありません。僕は前に穂高山はもちろん、槍ヶ岳にも登っていましたから、朝霧の下りた梓川の谷を案
Pred completed in 28.69 sec.
```
