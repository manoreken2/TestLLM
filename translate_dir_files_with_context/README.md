# LLM batch book translator with context 

Batch text translator using llamaindex and ollama LLM.
Translate all files on specified directory.

## Hardware requirements

1. 768GB main memory
2. No GPU required, runs on CPU

## run.py 起動方法

0. Enable developer mode to use symlinks https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

1. Download and install ollama. https://ollama.com/download

2. Then start command prompt and type

        ollama pull DeepSeek-R1:671b-0528-q8_0

3. Download and install Miniforge. The program is tested with Miniforge3-25.3.1-0. https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Windows-x86_64.exe

4. Download this TestLLM repository as zip and extract to C:\work as C:\work\TestLLM

5. Open miniforge3 prompt and type

        c:
        cd c:\work\TestLLM\translate_dir_files_with_context
        conda create -y -n Lidx python=3.11
        conda activate Lidx
        pip install triton-windows llama-index llama-index-llms-huggingface llama-index-readers-web pathlib markdown2 llama-index-embeddings-huggingface notebook ipywidgets widgetsnbextension pandas-profiling hf_xet qdrant_client transformers llama-index-llms-ollama
        pip install torch==2.10.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

Batch translates all text files on the specified folder.

        python run.py --input_dir=../shiji --output_file=shiji_translated.html

Translation output example:

Test input data 史記 download from project gutenberg
 https://github.com/manoreken2/TestLLM/tree/main/translate_dir_files_with_context/shiji

史記 batch translation output
 https://manoreken2.github.io/TestLLM/translate_dir_files_with_context/shiji.html

