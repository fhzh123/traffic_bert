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

from rnn import Encoder, Decoder, Seq2Seq
from dataset import CustomDataset, Transpose_tensor, getDataLoader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print('Data Loading...')
    start_time = time.time()

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
    encoder = Encoder(args.d_embedding, args.d_hidden, n_layers=args.n_layers, dropout=args.dropout)
    decoder = Decoder(args.d_embedding, args.d_hidden, n_layers=args.n_layers, dropout=args.dropout)
    seq2seq = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, seq2seq.parameters()), lr=1e-5, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.MSELoss()
    seq2seq.to(device)
    print(seq2seq)

    # Preparing
    best_val_loss = None
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(f'./save/save_rnn_{nowDatetime}'):
        os.mkdir(f'./save/save_rnn_{nowDatetime}')
    hyper_parameter_setting = dict()
    hyper_parameter_setting['data'] = 'total'
    hyper_parameter_setting['n_layers'] = args.n_layers
    hyper_parameter_setting['d_hidden'] = args.d_hidden
    hyper_parameter_setting['d_embedding'] = args.d_embedding
    with open(f'./save/save_rnn_{nowDatetime}/hyper_parameter_setting.txt', 'w') as f:
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
                seq2seq.train()
            if phase == 'valid':
                seq2seq.eval()
                val_loss = 0
            for stop_ix, (src, src_rev, trg) in enumerate(tqdm(dataloader_dict[phase])):

                if phase == 'train' and stop_ix == 30000:
                    break
                
                # Input to Device(CUDA) with float tensor
                src = src.float().to(device)
                trg = trg.float().to(device)

                # Optimizer Setting
                optimizer.zero_grad()

                # Model Training & Validation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = seq2seq(src, trg)
                    loss = criterion(outputs.squeeze(2), trg)
                    # Backpropagate Loss
                    if phase == 'train':
                        loss.backward()
                        torch_utils.clip_grad_norm_(seq2seq.parameters(), args.grad_clip)
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
                pd.DataFrame(total_loss_list).to_csv('./save/save_rnn_{}/{} epoch_loss.csv'.format(nowDatetime, e), index=False)
            if phase == 'valid': 
                print('='*45)
                val_loss /= len(dataloader_dict['valid'])
                print("[Epoch:%d] val_loss:%5.3f | spend_time:%5.2fmin"
                        % (e, val_loss, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    val_loss_save = round(val_loss, 2)
                    torch.save(model.state_dict(), f'./save/save_rnn_{nowDatetime}/model_{e}_{val_loss_save}.pt')
                    best_val_loss = val_loss

        # Gradient Scheduler Step
        scheduler.step()

    print('Done...!')

if __name__ == '__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Traffic-BERT Argparser')
    parser.add_argument('--data', type=str, default='pems', help='Set dataset; Default is pems')

    parser.add_argument('--num_epoch', type=int, default=6, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size; Default is 100')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate; Default is 1e-5')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--d_hidden', default=256, type=int, help='model dimension')
    parser.add_argument('--d_embedding', default=256, type=int, help='embedding dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Model layers; Default is 5')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Ratio; Default is 0.1')

    parser.add_argument('--print_freq', type=int, default=1000, help='Print train loss frequency; Default is 1000')
    args = parser.parse_args()

    main(args)