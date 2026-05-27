# jupyter chat Setup

```
llama-server \
	-m ~/hf/deepseek-v4-pro/DeepSeek-V4-Pro-Q4_K_M.gguf \
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

```
conda create -y -n jupyter python=3.12
conda activate jupyter
conda install pip
pip install openai notebook

```

```
conda activate jupyter
cd jupyter_chat
jupyter notebook Run.ipynb
```
