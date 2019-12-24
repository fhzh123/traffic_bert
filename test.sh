#! /bin/bash

python3 train.py --src_rev_usage=False --n_layers=12 --batch_size=200 --num_epoch=6 --data=pems --repeat_input=True
python3 train.py --src_rev_usage=False --n_layers=12 --batch_size=200 --num_epoch=6 --data=pems --repeat_input=True --src_rev_usage=False
python3 train.py --src_rev_usage=False --n_layers=24 --batch_size=100 --num_epoch=6 --data=pems --repeat_input=True