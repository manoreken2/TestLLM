# LLM book translator

long text translator using ollama LLM

# Hardware requirements

1. 32GB main memory
2. NVIDIA RTX 3090, 4090, or 5090

# Setup on Windows platforms

0. Enable developer mode to use symlinks https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

1. Download and install ollama. https://ollama.com/download

2. Then start command prompt and type

        ollama run deepseek-r1:32b

3. Download and install Miniforge. The program is tested with Miniforge3-25.3.0-3. https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Windows-x86_64.exe

4. Download this TestLLM repository as zip and extract to C:\work as C:\work\TestLLM

5. Open miniforge3 prompt and type

        c:
        cd c:\work\TestLLM\translate
        conda create -n TestLLM python=3.12
        conda activate TestLLM
        pip install -r requirements.txt

# Perform book translation

1. Open miniforge3 prompt and type 

        c:
        cd c:\work\TestLLM\translate
        conda activate TestLLM
        python translate2.py --input=inferno_SV.txt --tgt_lang="Modern English" --output=inferno_EN_32b.html







