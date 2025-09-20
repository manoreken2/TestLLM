GPT2 model Training experiment with Japanese text

program is derived from the book [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D).


## Prepare dataset

Converts Aozora bunko text to UTF-8

python 00_cleanup.py

Put UTF-8 encoded txt onto orig directory then run

python 01_prepare_dataset.py



## Train

set configuration params onto train_conf.yaml and run

python 02_train.py


