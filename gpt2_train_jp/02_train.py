import gc
import torch
import tiktoken
import time
import matplotlib.pyplot as plt
from importlib.metadata import version
from previous_chapters import create_dataloader_v1
from previous_chapters import GPTModel
from previous_chapters import plot_loss_perplexities
from train_model import load_model
from train_model import new_model
from train_model import train_model
import yaml

# 過学習かどうかを判断するためのグラフ。
def CreateGraphs(conf, train_losses, tokens_seen, val_losses):
    n_epochs = conf['epochs']
    epochs_tensor = torch.linspace(1, n_epochs, len(train_losses))
    plot_loss_perplexities(conf['name'], epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.tight_layout(); plt.savefig(f"Loss_{conf['name']}.pdf")
    #plt.show()

def Train(device, conf):
    print(conf)

    # 途中から再開するとき用。
    checkpoint_filename = conf['saved_checkpoint']

    # 訓練に使用する文章。
    with open(conf['train_txt'], "r", encoding="utf-8") as file:
        text_data = file.read()

    # テスト出力用プロンプト文章。
    with open(conf['test_txt'], "r", encoding="utf-8") as file:
        test_txt_list = [line.rstrip() for line in file]

    n_epochs = conf['epochs'] # 15
    checkpoint_count = 5 # 5
    checkpoint_epoch_interval = int(n_epochs / checkpoint_count)
    assert(0 == (n_epochs % checkpoint_epoch_interval))

    # learning rate config
    initial_lr = 0.0001
    peak_lr = 0.001  # this was originally set to 5e-4 in the book by mistake
    min_lr = initial_lr # 0.1 * initial_lr
    weight_decay = 0.1

    # "gpt2", "o200k_base"等。
    tokenizer = tiktoken.get_encoding(conf['tokenizer'])

    # https://storage.prod.researchhub.com/uploads/papers/2020/06/01/language-models.pdf
    with open('gpt2_conf_list.yaml', 'r', encoding='utf-8') as f:
        gpt2_conf_list = yaml.safe_load(f)
    gpt2_conf = gpt2_conf_list[ conf['model'] ]
    # 共通の設定値。
    gpt2_conf["vocab_size"] = tokenizer.n_vocab
    gpt2_conf["drop_rate"] = conf['drop_rate']
    gpt2_conf["qkv_bias"] = False
    gpt2_conf["context_length"] = 1024

    split_idx = int(conf['train_val_ratio'] * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        tokenizer,
        gpt2_conf,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        tokenizer,
        gpt2_conf,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps) # 20% warmup

    if len(checkpoint_filename) == 0:
        model, optimizer = new_model(device, gpt2_conf, peak_lr, weight_decay)
    else:
        model, optimizer = load_model(device, gpt2_conf, checkpoint_filename)

    train_losses, val_losses, tokens_seen, lrs = train_model(
        conf['name'],
        model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
        eval_iter=1, test_txt_list=test_txt_list,
        tokenizer=tokenizer, warmup_steps=warmup_steps, 
        initial_lr=initial_lr, min_lr=min_lr, checkpoint_epoch_interval=checkpoint_epoch_interval)

    # 過学習かどうかを判断するためのグラフ。
    CreateGraphs(conf, train_losses, tokens_seen, val_losses)

    del model
    del optimizer
    del tokenizer
    del train_loader
    del val_loader
    del text_data

def main():

    with open('train_conf.yaml', 'r', encoding='utf-8') as f:
        train_conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version {version("torch")}. Using {device} device")

    for conf in train_conf:
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.manual_seed(123)

        Train(device, conf)

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        print(f"Completed in {(time.time() - start_time):.2f} sec.")

if __name__ == "__main__":
    main()


