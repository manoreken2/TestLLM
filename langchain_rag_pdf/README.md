
Qwen RAG program to parse pdf and answer the user query.

# Setup

Install Miniforge. then type 
```
conda create -n LCrag -y python=3.11
conda activate LCrag
pip install transformers langchain faiss-cpu peft sentence-transformers unstructured pdfminer.six langchain_huggingface langchain-community langchain-core
```

# Run


Program runs on CPU not GPU

```
export CUDA_VISIBLE_DEVICES=""
conda activate LCrag
python run.py -pdf "saiyuuki.pdf" -q "How 悟空 flies? Does he use some equipment to fly?"
```

# Environment tested

```
pip list
Package                  Version
------------------------ ---------
accelerate               1.12.0
aiofiles                 25.1.0
aiohappyeyeballs         2.6.1
aiohttp                  3.13.3
aiosignal                1.4.0
annotated-doc            0.0.4
annotated-types          0.7.0
anyio                    4.12.1
attrs                    25.4.0
beautifulsoup4           4.14.3
blis                     1.3.3
catalogue                2.0.10
certifi                  2026.1.4
cffi                     2.0.0
charset-normalizer       3.4.4
click                    8.3.1
cloudpathlib             0.23.0
confection               0.1.5
cryptography             46.0.5
cuda-bindings            12.9.4
cuda-pathfinder          1.3.4
cymem                    2.0.13
dataclasses-json         0.6.7
emoji                    2.15.0
faiss-cpu                1.13.2
filelock                 3.24.3
filetype                 1.2.0
frozenlist               1.8.0
fsspec                   2026.2.0
greenlet                 3.3.2
h11                      0.16.0
hf-xet                   1.2.0
html5lib                 1.1
httpcore                 1.0.9
httpx                    0.28.1
httpx-sse                0.4.3
huggingface_hub          0.36.2
idna                     3.11
installer                0.7.0
Jinja2                   3.1.6
joblib                   1.5.3
jsonpatch                1.33
jsonpointer              3.0.0
langchain                1.2.10
langchain-classic        1.0.1
langchain-community      0.4.1
langchain-core           1.2.14
langchain-huggingface    1.2.0
langchain-text-splitters 1.1.1
langdetect               1.0.9
langgraph                1.0.9
langgraph-checkpoint     4.0.0
langgraph-prebuilt       1.0.8
langgraph-sdk            0.3.8
langsmith                0.7.6
llvmlite                 0.46.0
lxml                     6.0.2
markdown-it-py           4.0.0
MarkupSafe               3.0.3
marshmallow              3.26.2
mdurl                    0.1.2
mpmath                   1.3.0
multidict                6.7.1
murmurhash               1.0.15
mypy_extensions          1.1.0
networkx                 3.6.1
numba                    0.64.0
numpy                    2.4.2
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.27.5
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvshmem-cu12      3.4.5
nvidia-nvtx-cu12         12.8.90
olefile                  0.47
orjson                   3.11.7
ormsgpack                1.12.2
packaging                26.0
pdfminer.six             20260107
peft                     0.18.1
pillow                   12.1.1
pip                      26.0.1
preshed                  3.0.12
propcache                0.4.1
psutil                   7.2.2
pycparser                3.0
pydantic                 2.12.5
pydantic_core            2.41.5
pydantic-settings        2.13.1
Pygments                 2.19.2
pypdf                    6.7.2
pypdfium2                5.5.0
python-dotenv            1.2.1
python-iso639            2026.1.31
python-magic             0.4.27
python-oxmsg             0.0.2
PyYAML                   6.0.3
RapidFuzz                3.14.3
regex                    2026.2.19
requests                 2.32.5
requests-toolbelt        1.0.0
rich                     14.3.3
safetensors              0.7.0
scikit-learn             1.8.0
scipy                    1.17.1
sentence-transformers    5.1.2
setuptools               82.0.0
shellingham              1.5.4
six                      1.17.0
smart_open               7.5.0
soupsieve                2.8.3
spacy                    3.8.11
spacy-legacy             3.0.12
spacy-loggers            1.0.5
SQLAlchemy               2.0.46
srsly                    2.5.2
sympy                    1.14.0
tenacity                 9.1.4
thinc                    8.3.10
threadpoolctl            3.6.0
tokenizers               0
```
