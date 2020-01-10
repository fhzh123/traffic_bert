# Import Module
import os
import h5py
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

#from warmup_scheduler import GradualWarmupScheduler

# Import Custom Module
from autoencoder import *
from dataset import CustomDataset, Transpose_tensor, getDataLoader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print('Data Loading...')
    start_time = time.time()

    # if args.data == 'pems':
    #     train_data_path = './preprocessing/pems_preprocessed_train.h5'
    #     valid_data_path = './preprocessing/pems_preprocessed_valid.h5'
    # if args.data == 'metr':
    #     train_data_path = './preprocessing/metr_preprocessed_train.h5'
    #     valid_data_path = './preprocessing/metr_preprocessed_valid.h5'
    train_data_path = './preprocessing/total/preprocessed_train.h5'
    valid_data_path = './preprocessing/total/preprocessed_valid.h5'
        
    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

    print('DataLoader Setting...')

    dataset_dict = {
        'train': CustomDataset(train_data_path),
        'valid': CustomDataset(valid_data_path)
    }
    dataloader_dict = {
        'train': getDataLoader(dataset_dict['train'], args.batch_size, True, args.num_workers),
        'valid': getDataLoader(dataset_dict['valid'], args.batch_size, True, args.num_workers)
    }

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

    # Train & Valid & Test Split
    print('Model Setting...')
    layers = list(map(int, args.layers.split('-')))
    model = SAE(layers, args.dropout)
                    #    src_rev_usage=args.src_rev_usage, repeat_input=args.repeat_input)
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
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(f'./save/save_{nowDatetime}'):
        os.makedirs(f'./save/save_{nowDatetime}')
    hyper_parameter_setting = dict()
    hyper_parameter_setting['data'] = args.data
    hyper_parameter_setting['num_workers'] = args.num_workers
    hyper_parameter_setting['dropout'] = args.dropout
    hyper_parameter_setting['layers'] = args.layers
    
    with open(f'./save/save_{nowDatetime}/hyper_parameter_setting.txt', 'w') as f:
        for key in hyper_parameter_setting.keys():
            f.write(str(key) + ': ' + str(hyper_parameter_setting[key]))
            f.write('\n')

    # Model Train
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
            for src, _, trg in tqdm(dataloader_dict[phase]):
                # Input to Device(CUDA) with float tensor
                src = src.float().to(device)
                trg = trg.float().to(device)

                # Optimizer Setting
                optimizer.zero_grad()

                # Model Training & Validation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(src)
                    loss = criterion(outputs, trg)
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
                pd.DataFrame(total_loss_list).to_csv('./save/save_{}/{} epoch_loss.csv'.format(nowDatetime, e), index=False)
            if phase == 'valid': 
                print('='*45)
                val_loss /= len(dataloader_dict['valid'])
                print("[Epoch:%d] val_loss:%5.3f | spend_time:%5.2fmin"
                        % (e, val_loss, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    val_loss_save = round(val_loss, 2)
                    torch.save(model.state_dict(), f'./save/save_{nowDatetime}/model_{e}_{val_loss_save}.pt')
                    best_val_loss = val_loss

        # Gradient Scheduler Step
        scheduler.step()

    print('Done...!')

if __name__ == '__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Traffic-BERT Argparser')
    parser.add_argument('--data', type=str, default='pems', help='Set dataset; Default is pems')
    parser.add_argument('--num_workers', type=int, default=0, help='')

    parser.add_argument('--num_epoch', type=int, default=6, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size; Default is 100')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate; Default is 1e-5')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Ratio; Default is 0.5')
    parser.add_argument('--layers', default='12-300-300-300-12', help='list of stacked autoencoder layer dimensions')

    parser.add_argument('--print_freq', type=int, default=10000, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)