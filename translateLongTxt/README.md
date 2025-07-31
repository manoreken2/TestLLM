# LLM book translator

Long text translator using ollama LLM generative AI

## Hardware requirements

1. 32GB main memory
2. NVIDIA RTX 3090, 4090, or 5090

## Setup on Windows platforms

0. Enable developer mode to use symlinks https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

1. Download and install ollama. https://ollama.com/download

2. Then start command prompt and type

        ollama run deepseek-r1:32b

3. Download and install Miniforge. The program is tested with Miniforge3-25.3.1-0. https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Windows-x86_64.exe

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
        python translateLongTxt.py --input=test/inferno_SV.txt --tgt_lang="Modern English" --output=inferno_EN_32b.html

Translation input example: https://github.com/manoreken2/TestLLM/blob/main/translateLongTxt/test/inferno_SV.txt
Translation output example: https://manoreken2.github.io/TestLLM/translateLongTxt/test/Strindberg_Inferno_JP_DeepSeekR1_671b_0528_q8.html


