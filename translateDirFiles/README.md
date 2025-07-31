# LLM batch book translator

Batch text translator using ollama LLM.
Translate all files on specified directory.

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

# TranslateDirFiles

Batch translates all text files on the specified folder.

        python translateDirFiles.py --input_dir=sikyou --tgt_lang=`Modern Japanese with commentary` --model_name=DeepSeek-R1:671b-0528-q4_K_M --output_file=sikyou_translated.html

Translation output example:

Test input data 詩経
 https://manoreken2.github.io/TestLLM/translateDirFiles/sikyou

詩経 batch translation output
 https://manoreken2.github.io/TestLLM/translateDirFiles/sikyou_671b_0528_q4.html
詩経 summary
 https://manoreken2.github.io/TestLLM/translateDirFiles/sikyou_summary.html



