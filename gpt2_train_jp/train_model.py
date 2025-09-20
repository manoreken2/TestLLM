import math
from previous_chapters import evaluate_model, generate_and_print_sample
from previous_chapters import calc_loss_batch
from gpt_model_fast import GPTModelFast
import torch
import os


def new_model(device, config, peak_lr, weight_decay):
    model = GPTModelFast(config).bfloat16()
    #model = GPTModel(config).bfloat16()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)  # the book accidentally omitted the lr assignment
    return model, optimizer

def load_model(device, config, path):
    checkpoint = torch.load(path, weights_only=True)
    model = GPTModelFast(config).bfloat16()
    #model = GPTModel(config).bfloat16()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer

def create_dir(path):
    dir = os.path.dirname(path)
    if 0 < len(dir) and not os.path.exists(dir):
        os.makedirs(dir)

def save_model_opt(model, optimizer, path):
    create_dir(path)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        path)


def train_model(name, model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_iter, test_txt_list, tokenizer,
                warmup_steps, initial_lr, min_lr, checkpoint_epoch_interval, test_output_tokens=100):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

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
            print(f"{name} Ep {epoch+1} (Iter {global_step:06d}) lr={lr:.3e}: "
                    f"Train loss {train_loss:.3e}, perp={math.exp(train_loss):.3f}, "
                    f"Val loss {val_loss:.3f}, perp={math.exp(val_loss):.1f}")

    # モデルのパラメーターを保存します。
    save_model_opt(model, optimizer, f"chkpt_{name}/ep{epoch+1}.pth")

    # Generate and print a sample from the model to monitor progress
    with open(f'Predicts_{name}.txt', 'w', encoding='utf-8') as f:
        for test_txt in test_txt_list:
            generate_and_print_sample(f, model, tokenizer, device, test_txt, test_output_tokens)

    return train_losses, val_losses, track_tokens_seen, track_lrs

