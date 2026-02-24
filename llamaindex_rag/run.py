'''
https://qwen.readthedocs.io/en/latest/framework/LlamaIndex.html

conda create -y -n Lidx python=3.11
conda activate Lidx
pip install llama-index
pip install llama-index-llms-huggingface
pip install llama-index-readers-web
pip install llama-index-embeddings-huggingface

'''

import torch
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set prompt template for generation (optional)
from llama_index.core import PromptTemplate

def completion_to_prompt(completion):
   return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt

# Set Qwen2.5 as the language model and set generation config
Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen3-235B-A22B-Instruct-2507", # "Qwen/Qwen2.5-7B-Instruct",
    tokenizer_name="Qwen/Qwen3-235B-A22B-Instruct-2507", # "Qwen/Qwen2.5-7B-Instruct",
    context_window=30000,
    max_new_tokens=2000,
    generate_kwargs={"temperature": 0.7, "top_k": 20, "top_p": 0.8}, # {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-base-en-v1.5"
)

# Set the size of the text chunk for retrieval
Settings.transformations = [SentenceSplitter(chunk_size=1024)]

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["../document/saiyuuki.txt"]).load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations
)

query_engine = index.as_query_engine()
your_query = "How 悟空 flies? Does he use some equipment to fly?"
print(query_engine.query(your_query).response)

