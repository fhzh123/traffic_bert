#! /bin/bash

python3 train.py --src_rev_usage=False --n_layers=6 --batch_size=400 --num_epoch=5 --data=metr
python3 train.py --src_rev_usage=True --n_layers=6 --batch_size=400 --num_epoch=5 --data=metr
python3 train.py --src_rev_usage=True --n_layers=12 --batch_size=300 --num_epoch=5 --data=metr
python3 train.py --src_rev_usage=False --n_layers=12 --batch_size=300 --num_epoch=5 --data=metr