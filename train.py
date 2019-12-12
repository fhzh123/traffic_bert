# Import Module
import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

from warmup_scheduler import GradualWarmupScheduler

# Import Custom Module
from bert import littleBERT
from dataset import CustomDataset, Transpose_tensor, getDataLoader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print('Data Loading...')
    start_time = time.time()

    with h5py.File(args.train_data_path, 'r') as f:
        # List all groups
        src = list(f.keys())[0] # Previous data
        trg = list(f.keys())[1] # After data

        pems_src_train = list(f[src])
        pems_trg_train = list(f[trg])

    with h5py.File(args.valid_data_path, 'r') as f:
        # List all groups
        src = list(f.keys())[0] # Previous data
        trg = list(f.keys())[1] # After data

        pems_src_valid = list(f[src])
        pems_trg_valid = list(f[trg])
        
    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

    print('DataLoader Setting...')

    dataset_dict = {
        'train': CustomDataset(src=pems_src_train, trg=pems_trg_train),
        'valid': CustomDataset(src=pems_src_valid, trg=pems_trg_valid)
    }
    dataloader_dict = {
        'train': getDataLoader(dataset_dict['train'], args.batch_size, True),
        'valid': getDataLoader(dataset_dict['valid'], args.batch_size, True)
    }

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

    # Train & Valid & Test Split
    print('Model Setting...')

    model = littleBERT(n_head=args.n_head, d_model=args.d_model, d_embedding=args.d_embedding, 
                       n_layers=args.n_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    #scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch)
    #scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler_cosine)
    criterion = nn.MSELoss()
    model.to(device)

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

    # Preparing
    best_val_loss = None
    if not os.path.exists('./save'):
        os.mkdir('./save')

    #
    for e in range(args.num_epoch):
        start_time_e = time.time()
        for phase in ['train', 'valid']:
            # Preparing
            total_loss_list = list()
            freq = args.print_freq - 1
            # Model Setting
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_loss = 0
            for src, src_rev, trg in tqdm(dataloader_dict[phase]):
                # Input to Device(CUDA) with float tensor
                src = src.float().to(device)
                src_rev = src_rev.float().to(device)
                trg = trg.float().to(device)

                # Optimizer Setting
                optimizer.zero_grad()

                # Model Training & Validation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(src, src_rev)
                    loss = criterion(outputs.squeeze(2), trg)
                    # Backpropagate Loss
                    if phase == 'train':
                        loss.backward()
                        torch_utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                        # Print every setted frequency
                        freq += 1
                        if freq == args.print_freq:
                            total_loss = loss.item()
                            print("[loss:%5.2f]" % (total_loss))
                            total_loss_list.append(total_loss)
                            freq = 0
                    if phase == 'valid':
                        val_loss += loss.item()

            # Finishing iteration
            if phase == 'train':
                pd.DataFrame(total_loss_list).to_csv('./save/{} epoch_loss.csv'.format(e), index=False)
            if phase == 'valid': 
                print('='*45)
                val_loss /= len(dataloader_dict['valid'])
                print("[Epoch:%d] val_loss:%5.3f | spend_time:%5.2fmin"
                        % (e, val_loss, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    torch.save(model.state_dict(), './save/model_{}.pt'.format(e))
                    best_val_loss = val_loss

        # Gradient Scheduler Step
        scheduler.step()

    print('Done...!')

if __name__ == '__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Traffic-BERT Argparser')
    parser.add_argument('--train_data_path', 
        default='./preprocessing/pems_preprocessed_train.h5', 
        type=str, help='path of data h5 file (train)')
    parser.add_argument('--valid_data_path', 
        default='./preprocessing/pems_preprocessed_valid.h5', 
        type=str, help='path of data h5 file (valid)')

    parser.add_argument('--num_epoch', type=int, default=5, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size; Default is 100')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate; Default is 1e-5')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=1, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--d_model', default=768, type=int, help='model dimension')
    parser.add_argument('--d_embedding', default=256, type=int, help='embedding dimension')
    parser.add_argument('--n_head', default=12, type=int, help='number of head in self-attention')
    parser.add_argument('--dim_feedforward', default=768*4, type=int, help='dimension of feedforward net')
    parser.add_argument('--n_layers', type=int, default=12, help='Model layers; Default is 5')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Ratio; Default is 0.1')

    parser.add_argument('--print_freq', type=int, default=500, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)