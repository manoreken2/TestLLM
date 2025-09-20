import torch
import tiktoken
import time
from importlib.metadata import version
from train_model import load_model
from train_model import generate_and_print_sample
import yaml

start_time = time.time()

torch.manual_seed(123)

checkpoint_filename = "checkpoints_Small_drop01/model_ep500.pth" 

tokenizer = tiktoken.get_encoding("gpt2")

with open('gpt2_conf_list.yaml', 'r') as f:
    gpt2_conf_list = yaml.safe_load(f)

cfg = gpt2_conf_list["GPT2_Small_conf"]

# 共通の設定値。
cfg["vocab_size"] = tokenizer.n_vocab
cfg["drop_rate"] = 0.0 # predのときはdropしない。
cfg["qkv_bias"] = False
cfg["context_length"] = 1024

with open('gpt2_conf_list.yaml', 'r') as f:
    gpt2_conf_list = yaml.safe_load(f)

cfg = gpt2_conf_list["GPT2_Small_conf"]

# 共通の設定値。
cfg["vocab_size"] = tokenizer.n_vocab
cfg["drop_rate"] = 0.0 # pred時はdropしない。
cfg["qkv_bias"] = False
cfg["context_length"] = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch version:", version("torch"))
print(f"Using {device} device.")

model, optimizer = load_model(device, cfg, checkpoint_filename)

generate_and_print_sample(model, tokenizer, device, "河童")

end_time = time.time()
print(f"Pred completed in {(end_time - start_time):.2f} sec.")


