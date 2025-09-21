import gc
import torch
from helpers import create_dataloader_v1, setup_torch_device
from train_model import new_model, new_tokenizer, train_model, load_gpt2_conf, load_train_conf_list


def Train(device, train_conf):

    # 訓練に使用する文章。
    with open(train_conf['train_txt'], "r", encoding="utf-8") as file:
        text_data = file.read()

    # テスト出力用プロンプト文章。
    with open(train_conf['test_txt'], "r", encoding="utf-8") as file:
        test_txt_list = [line.rstrip() for line in file]

    # 訓練エポック数関連設定。
    n_epochs = train_conf['epochs'] # 15
    checkpoint_count = train_conf['checkpoint_count'] # 5
    checkpoint_epoch_interval = int(n_epochs / checkpoint_count)
    assert(0 == (n_epochs % checkpoint_epoch_interval))

    # GPT2Tokenizer。"gpt2", "o200k_base"等。
    tokenizer = new_tokenizer(train_conf)

    # GPT2の種類。Small, Medium, Large, XL
    gpt2_conf = load_gpt2_conf('gpt2_conf_list.yaml', train_conf, tokenizer.n_vocab)

    # 訓練データとValデータ作成。
    split_idx = int(train_conf['train_val_ratio'] * len(text_data))
    train_loader = create_dataloader_v1(text_data[:split_idx], tokenizer, gpt2_conf, drop_last=True,  shuffle=True,  num_workers=0)
    val_loader   = create_dataloader_v1(text_data[split_idx:], tokenizer, gpt2_conf, drop_last=False, shuffle=False, num_workers=0)

    # 訓練するモデルとOptimizerを作成。
    model, optimizer = new_model(device, train_conf, gpt2_conf)

    # 訓練実行。
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps) # 20% warmup
    train_model(
        train_conf, gpt2_conf, model, train_loader, val_loader, optimizer, device,
        n_epochs=n_epochs, eval_iter=1, test_txt_list=test_txt_list,
        tokenizer=tokenizer, warmup_steps=warmup_steps,
        checkpoint_epoch_interval=checkpoint_epoch_interval)

    del model
    del optimizer
    del tokenizer
    del train_loader
    del val_loader
    del text_data


def main():

    device = setup_torch_device()

    # Train loop
    train_conf_list = load_train_conf_list()

    for train_conf in train_conf_list:
        print(train_conf)

        torch.cuda.empty_cache()
        torch.manual_seed(123)

        Train(device, train_conf)

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


