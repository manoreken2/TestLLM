# DeepSeek-V3.1 TerminusをパソコンでCPU実行

## 概要

llama.cppを使用し、DeepSeek-V3.14-ProのQ4_K_MクオンタイズドモデルをCPUで動かします。安定動作します。

参考ページ

https://unsloth.ai/docs/models/tutorials/deepseek-v3.1-how-to-run-locally


## 準備

メモリを768 GB積んだintel PCに
Ubuntu 24.04 LTSをインストールします。
SSDが2TB程度必要です。

TechPowerupの「Features」表等で使用CPUの対応命令セットを調べます。
https://www.techpowerup.com/cpu-specs/xeon-e5-2699a-v4.c3227

UbuntuのTerminalを開き、Miniforgeをダウンロードしてインストールします。
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ./Miniforge3-Linux-x86_64.sh
```
## llama.cppをダウンロードし、ビルド



```
# C++ビルド環境のUbuntuパッケージのインストール

cd ~/;
sudo apt update
sudo apt install -y git build-essential cmake ninja-build ccache curl \
libopenblas-dev libcurl4-openssl-dev libssl-dev \
numactl hwloc htop pkg-config

# llama.cppダウンロード

git clone https://github.com/ggml-org/llama.cpp

# llama.cppビルド。使用CPUの対応命令セットにONと指定。

cd llama.cpp
cmake -S . -B build  \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=OFF \
-DLLAMA_CURL=ON \
-DGGML_AVX=ON \
-DGGML_AVX2=ON \
-DGGML_FMA=ON \
-DGGML_F16C=ON \
-DGGML_BMI2=ON \
-DGGML_AVX512=OFF \
-DGGML_AMX_TILE=OFF \
-DGGML_AMX_INT8=OFF \
-DGGML_AMX_BF16=OFF \
-DGGML_BLAS=ON \
-DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release -j --clean-first --target llama-quantize llama-cli \
llama-gguf-split llama-mtmd-cli llama-server
```

## パラメーターファイルのダウンロード

huggingface: https://huggingface.co/

huggingfaceのアカウントを作成し(無料)、Access Tokenを生成します。Access Tokenはhf_で始まる38文字の文字列です。以下の様にしてパラメーターファイルをダウンロード。

```
# condaの"hfenv"環境作成。
conda create -y -n hfenv python=3.12
conda activate hfenv
conda install pip
pip install -U huggingface_hub hf_transfer

# パラメーターファイルのダウンロード。
hf download unsloth/DeepSeek-V3.1-Terminus-GGUF --include "*Q8_0*" --local-dir ~/hf \
--max-workers 1 --token hf_************

# ダウンロードしたshardをmergeして1個のggufファイルを作る。
 cd ~/llama.cpp
./build/bin/llama-gguf-split --merge \
 ~/hf/Q8_0/*Q8_0-00001-of-00015.gguf  \
 ~/hf/DeepSeek-V3.1-Terminus-Q8_0.gguf

# shardを削除。
rm ~/hf/Q8_0/*of-00021.gguf
```

## テスト実行

```
cd ~/llama.cpp
numactl --interleave=all ./build/bin/llama-cli \
	--model ~/hf/DeepSeek-V3.1-Terminus-Q8_0.gguf \
--prompt "Hello. Please answer with OK only." \
 --jinja \
 --threads-batch 10 \
 --temp 0.6 \
 --top-p 0.95 \
 --min-p 0.01 \
 --ctx-size 4096 \
 --predict 4096 \
 --parallel 1 \
 -n 10
```
正常動作します。

## サーバープロセス起動

```
cd ~/llama.cpp
numactl --interleave=all ./build/bin/llama-server \
--model ~/hf/DeepSeek-V3.1-Terminus-Q8_0.gguf \
 --jinja \
 -t 10 \
 -n 10 \
 --parallel 1 \
 --temp 0.6 \
 --top-p 0.95 \
 --min-p 0.01 \
 --ctx-size 4096 \
 --predict 4096 \
 --host 0.0.0.0 \
--port 8080 \
--api-key "a"
```


## サーバープロセスに接続しチャットするプログラム実行

チャット実行環境作成。
```
conda create -y -n jupyter_chat python=3.12
conda activate jupyter_chat
conda install pip
pip install IPython ipywidgets notebook
```
チャットプログラムのダウンロード、実行。
サーバーを実行したPCと同じPC上で実行する場合ipynbは無変更で動作します。別のPCで動かすときipynbのIPアドレスを変更して動かして下さい。

```
cd ~
git clone https://github.com/manoreken2/TestLLM.git
conda activate jupyter_chat
cd ~/TestLLM/jupyter_chat
jupyter notebook JupyterChat_DeepSeekV31Terminus.ipynb
```
