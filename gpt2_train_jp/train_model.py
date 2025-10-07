import time
import math
from helpers import evaluate_model, generate_and_print_sample
from helpers import calc_loss_batch
from helpers import create_loss_graph
from gpt_model_fast import GPTModelFast
import torch
import os
import tiktoken
import yaml


def load_train_conf_list():
    with open('train_conf.yaml', 'r', encoding='utf-8') as f:
        train_conf_list = yaml.safe_load(f)

    for train_conf in train_conf_list:
        train_conf.setdefault('initial_lr', 0.0001)
        train_conf.setdefault('peak_lr', 0.001)
        train_conf.setdefault('drop_rate', 0.1)
        train_conf.setdefault('train_val_ratio', 0.9)
        # learning rate config
        train_conf.setdefault('min_lr', train_conf['initial_lr'])
        train_conf.setdefault('weight_decay', 0.1)

    return train_conf_list


def load_gpt2_conf(path, train_conf, n_vocab):
    with open('gpt2_conf_list.yaml', 'r', encoding='utf-8') as f:
        gpt2_conf_list = yaml.safe_load(f)

    gpt2_conf = gpt2_conf_list[ train_conf['model'] ]

    # 共通の設定値セット。
    gpt2_conf.setdefault("vocab_size", n_vocab)
    gpt2_conf.setdefault("drop_rate", train_conf['drop_rate'])
    gpt2_conf.setdefault("qkv_bias", False)
    gpt2_conf.setdefault("context_length", 1024)

    return gpt2_conf


# GPT2Tokenizer。"gpt2", "o200k_base"等。
def new_tokenizer(train_conf):
    tokenizer = tiktoken.get_encoding(train_conf['tokenizer'])
    return tokenizer


def create_dir(path):
    dir = os.path.dirname(path)
    if 0 < len(dir) and not os.path.exists(dir):
        os.makedirs(dir)


def new_model(device, train_conf, gpt2_conf):
    model = GPTModelFast(gpt2_conf)
    model = torch.compile(model)
    model.to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_conf['peak_lr'], weight_decay=train_conf['weight_decay'], fused=True)
    return model, optimizer


def load_model(device, checkpoint_filename):
    saved = torch.load(checkpoint_filename, weights_only=True)
    train_conf = saved['train_conf']
    gpt2_conf = saved['gpt2_conf']

    model = GPTModelFast(gpt2_conf)
    model.load_state_dict(saved["model_state_dict"])
    model = torch.compile(model)
    model.to(device).to(torch.bfloat16)

    tokenizer = tiktoken.get_encoding(train_conf['tokenizer'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_conf['peak_lr'], weight_decay=train_conf['weight_decay'], fused=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    return model, tokenizer, optimizer, train_conf, gpt2_conf


def save_model_opt(model, optimizer, train_conf, gpt2_conf, path):
    create_dir(path)

    compiled = hasattr(model, "_orig_mod")
    if compiled:
        model_to_save = model._orig_mod.state_dict()
    else:
        model_to_save = model.state_dict()

    torch.save({ "model_state_dict": model_to_save, "optimizer_state_dict": optimizer.state_dict(), "train_conf": train_conf, "gpt2_conf": gpt2_conf }, path)


def train_model(train_conf, gpt2_conf, model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_iter, test_txt_list, tokenizer,
                warmup_steps, checkpoint_epoch_interval, test_output_tokens=100):
    name=train_conf['name']
    initial_lr=train_conf['initial_lr']
    min_lr=train_conf['min_lr']

    start_time = time.time()

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    log_texts = ""

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # Calculate and backpropagate the loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping after the warmup phase to avoid exploding gradients
            if global_step >= warmup_steps:  # the book originally used global_step > warmup_steps, which led to a skipped clipping step after warmup
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            tokens_seen += input_batch.numel()

        if 0 == (epoch+1) % checkpoint_epoch_interval:
            # Periodically evaluate the model on the training and validation sets
            train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)

            # Print the current losses
            log_txt = f"{name} Ep {epoch+1} (Iter {global_step:06d}) lr={lr:.3e}: Train loss {train_loss:.3e}, perp={math.exp(train_loss):.3f}, Val loss {val_loss:.3f}, perp={math.exp(val_loss):.1f}"
            print(log_txt)
            log_texts += log_txt + "\n"

    # モデルのパラメーターを保存します。
    save_model_opt(model, optimizer, train_conf, gpt2_conf, f"chkpt_{name}/ep{epoch+1}.pth")

    # Generate and print a sample from the model to monitor progress
    with open(f'Predicts_{name}.txt', 'w', encoding='utf-8') as f:
        for test_txt in test_txt_list:
            generate_and_print_sample(model, tokenizer, device, test_txt, test_output_tokens, f)

    # 過学習かどうかを判断するためのグラフ。
    create_loss_graph(train_conf, train_losses, track_tokens_seen, val_losses)

    # ログ出力.
    elapsed_time_txt = f"Elapsed time: {(time.time() - start_time):.2f} sec."
    print(elapsed_time_txt)
    with open(f'Log_{name}.txt', 'w', encoding='utf-8') as f:
        f.write(f"{train_conf}\n{log_texts}{elapsed_time_txt}\n")




