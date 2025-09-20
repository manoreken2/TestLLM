import torch
import tiktoken
import time
from importlib.metadata import version
from train_model import load_model
from train_model import generate_and_print_sample
import yaml

start_time = time.time()

checkpoint_filename = "checkpoints_Small_drop01/model_ep500.pth" 

torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("僕は", tokenizer).to(device),
    max_new_tokens=300,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print(token_ids_to_text(token_ids, tokenizer))
