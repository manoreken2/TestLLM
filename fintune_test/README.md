Setup

requirements.txt is arranged for RTX5090.
see the following page and install triton windows for your GPU
https://github.com/woct0rdho/triton-windows

tested on WSL 2 Ubuntu 24.04

install miniforge

open bash prompt and type

    # install ollama onto WSL Ubuntu Linux 24.04 
    curl -fsSL https://ollama.com/install.sh | sh

    # used by model.save_pretrained_gguf
    sudo apt-get install libcurl4-openssl-dev cmake -y
    
    conda create -n Finetune python=3.12
    conda activate Finetune

    pip install -r requirements.txt

    # https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-wsl-ubuntu-13-0-local_13.0.2-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-13-0-local_13.0.2-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-13-0

    # PATHに/usr/local/cuda/binを追加

    # First uninstall xformers installed by previous libraries
    pip uninstall xformers -y

    # install pytorch with 13.0
    pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 --index-url https://download.pytorch.org/whl/cu130

    # Clone and build xformers
    pip install ninja
    export TORCH_CUDA_ARCH_LIST="12.0"
    git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
    cd xformers && python setup.py install && cd ..

Llama3_8B_Ollama.ipynb is downloaded from
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb#scrollTo=MwEbRFl0Mf3E

