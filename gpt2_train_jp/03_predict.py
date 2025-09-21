import torch
import time
from train_model import load_model
from train_model import generate_and_print_sample
from helpers import setup_torch_device
import argparse
import pathlib


def main(args):
    start_time = time.time()

    torch.manual_seed(123)
    device = setup_torch_device()

    model, tokenizer, _, _, _ = load_model(device, args.checkpoint_file)

    if args.input_text_file.suffix == '.txt':
        # テスト出力用プロンプト文章。
        with open(args.input_text_file, "r", encoding="utf-8") as file:
            txt_list = [line.rstrip() for line in file]
        for txt in txt_list:
            generate_and_print_sample(model, tokenizer, device, txt, args.max_new_tokens)
    else:
        generate_and_print_sample(model, tokenizer, device, args.input_str, args.max_new_tokens)

    end_time = time.time()
    print(f"Pred completed in {(end_time - start_time):.2f} sec.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=pathlib.Path, default="chkpt_Kappa_Small_o200k_ep50_drop0_LR1e3/ep50.pth")
    parser.add_argument('--input_str', type=str, default="僕は")
    parser.add_argument('--input_text_file', type=pathlib.Path, default="")
    parser.add_argument('--max_new_tokens', type=int, default=100)
    args = parser.parse_args()

    main(args)


