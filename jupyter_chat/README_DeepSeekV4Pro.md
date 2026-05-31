# DeepSeek-V4-ProをパソコンでCPU実行(WIP)

## 概要

bati.cppを使用し、DeepSeek-V4-ProのQ4_K_MクオンタイズドモデルをCPUで動かします。ベータ版的な物であり、動作不安定です。

参考ページ: https://github.com/batiai/bati.cpp

## 準備

メモリを1536 GB積んだintel PCに
Ubuntu 24.04 LTSをインストールします。
SSDが4TB程度必要です。

TechPowerupの「Features」表等で使用CPUの対応命令セットを調べます。
https://www.techpowerup.com/cpu-specs/xeon-e5-2699a-v4.c3227

UbuntuのTerminalを開き、Miniforgeをダウンロードしてインストールします。
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ./Miniforge3-Linux-x86_64.sh
```
## bati.cppをダウンロードし、ビルド

```
# C++ビルド環境のUbuntuパッケージのインストール
sudo apt update
sudo apt install -y git build-essential cmake ninja-build ccache \
libopenblas-dev libcurl4-openssl-dev libssl-dev \
numactl hwloc htop pkg-config

# bati.cppダウンロード
cd ~
git clone https://github.com/batiai/bati.cpp.git
cd bati.cpp
git tag -l 'v0.1.*'
git checkout v0.1.2 || true

# bati.cppビルド。使用CPUの対応命令セットにONと指定。
cmake -S . -B build -G Ninja \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_BUILD_TYPE=Release \
-DGGML_NATIVE=OFF \
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

cmake --build build --config Release -j 30 \
--target llama-cli llama-server llama-gguf-split llama-bench
```

## パラメーターファイルのダウンロード

huggingface: https://huggingface.co/

huggingfaceのアカウントを作成し(無料)、Access Tokenを生成します。Access Tokenはhf_で始まる38文字の文字列です。以下の様にしてパラメーターファイルをダウンロード。

```
# condaの"hfenv"環境作成。
conda create -y -n hfenv python=3.12
conda activate hfenv
pip install -U pip
pip install -U huggingface_hub hf_transfer

# パラメーターファイルのダウンロード。
hf download batiai/DeepSeek-V4-Pro-GGUF --include "*Q4_K_M*" \
--local-dir ~/hf/ --token hf_************* --max-workers 1

# ダウンロードしたshardをmergeして1個のggufファイルを作る。
cd ~/bati.cpp
./build/bin/llama-gguf-split --merge \
 ~/hf/deepseek-ai-DeepSeek-V4-Pro-Q4_K_M-00001-of-00021.gguf  \
 ~/hf/DeepSeek-V4-Pro-Q4_K_M.gguf

# shardを削除。
rm ~/hf/*of-00021.gguf
```

## テスト実行

```
cd ~/bati.cpp
numactl --interleave=all ./build/bin/llama-cli \
	--model ~/hf/DeepSeek-V4-Pro-Q4_K_M.gguf  \
--prompt "Hello. Please answer with OK only." \
 --jinja \
 --threads-batch 10 \
 --batch-size 32 \
 --ubatch-size 32 \
 --temp 0.6 \
 --top-p 0.95 \
 --min-p 0.01 \
 --ctx-size 256 \
 --predict 256 \
 --parallel 1 
```
出力のフォーマッティングが若干変ですが、動作します。

## サーバープロセス起動

```
cd ~/bati.cpp
numactl --interleave=all ./build/bin/llama-server \
	-m ~/hf/DeepSeek-V4-Pro-Q4_K_M.gguf \
-c 4096 \
-b 32 \
-ub 32 \
-t 10 \
-n 10 \
--parallel 1 \
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
サーバーを実行したPCと同じPC上で実行する場合Run.ipynbは無変更で動作します。別のPCで動かすときRun.ipynbのIPアドレスを変更して動かして下さい。

出力が異常の場合、jupyter kernelをrestartしてチャットプログラムを再実行して下さい。

```
cd ~
git clone https://github.com/manoreken2/TestLLM.git
conda activate jupyter_chat
cd ~/TestLLM/jupyter_chat
jupyter notebook JupyterChat_DeepSeekV4Pro.ipynb
```
