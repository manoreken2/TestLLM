# LLM batch book translator with context 

Batch text translator using llamaindex and llama.cpp.

Translate all files on specified directory.

## Hardware requirements

1. 768GB main memory
2. No GPU required, runs on CPU
3. C:\ drive capacity need 2TB or more

## run.py 起動方法

0. Enable developer mode to use symlinks https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

1. Install llama.cpp.

        winget install llama.cpp

2. Download and install Miniforge. The program is tested with Miniforge3-25.3.1-0. https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Windows-x86_64.exe

4. Download this TestLLM repository as zip and extract to C:\work as C:\work\TestLLM

5. Open Miniforge3 prompt and type the following commands to create hf_download conda environment.

        cd /D c:\work\TestLLM\translate_dir_files_with_context
        conda create -y -n hf_download python=3.12
        conda activate hf_download
        conda install -y pip
        pip install -U pip
        pip install -U huggingface_hub hf_transfer

5. Download GLM 5.2 GGUF parameter file.
Visit https://huggingface.co/ and sign in, Create user access token, it is free of charge. Token is text string beginning with hf.

        On Miniforge3 prompt, type the following commands.
        conda activate hf_download
        hf download unsloth/GLM-5.2-GGUF --local-dir unsloth/GLM-5.2-GGUF --include "*UD-Q6_K_XL*" --token hf_YOUR_USE_ACCESS_TOKEN_HERE --max-workers 1

        mkdir -p C:\work\hf
        llama-gguf-split --merge unsloth/GLM-5.2-GGUF/UD-Q6_K_XL/GLM-5.2-UD-Q6_K_XL-00001-of-00016.gguf C:/work/hf/GLM-5.2-UD-Q6_K_XL.gguf

        rmdir /s /q unsloth

5. Test run GLM 5.2 model on Lllama

        llama-cli -m C:/work/hf/GLM-5.2-UD-Q6_K_XL.gguf -p "Hello. Please answer with OK only." --temp 1.0 --top-p 0.95 --min-p 0.01 -c 4096 -t 10 --reasoning off

6. Start Llama server on your PC.

        llama-server -m C:/work/hf/GLM-5.2-UD-Q6_K_XL.gguf -c 4096 -t 10 --parallel 1 --host 0.0.0.0 --port 8080 --api-key "a"

7. Setup conda environment for the translation program.

        Open another Miniforge prompt.
        cd /D c:\work\TestLLM\translate_dir_files_with_context
        conda create -y -n Lidx2 python=3.12
        conda activate Lidx2
        conda install pip
        pip install llama-index llama-index-llms-openai-like openai pathlib markdown2

8. Start translation program on the same PC

        On miniforge prompt, type the following commands.
        cd /D c:\work\TestLLM\translate_dir_files_with_context
        conda activate Lidx2
        python run.py --input_dir=../document/Dante_La_Divina_Comedia --no-think --model_name=GLM_5.2_Q6_XL --paragraph_separate=True --context_window=4096 --output_file=s_dante_GLM52q6.html --tgt_lang=現代日本語 --extra_prompt=これはダンテ「神曲」の一節です。解説に文学的修辞技法、著者との関係、執筆当時の時代背景、古代ギリシャ・ローマ・聖書（書名 章:節）等西洋の古典からの引用、後世への影響があれば書いてください。


