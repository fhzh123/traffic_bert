# Import Module
import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(args):
    start_time = time.time()

    if args.processing_data == 'pems':
        data = pd.read_hdf(args.pems_filename) # 03/12 AM02:00 is missing!!
    if args.processing_data == 'metr':
        data = pd.read_hdf(args.metr_filename)

    print('data shape: ' + str(data.shape))

    prev_data = list()
    after_data = list()
    for col in tqdm(data.columns):
        for i in range(data.shape[0] - 24):
            input_ = data[col].iloc[i:i+12].tolist()
            output_ = data[col].iloc[i+12:i+24].tolist()
            if 0 not in input_ and 0 not in output_:
                prev_data.append(input_)
                after_data.append(output_)

    # Train & Valid & Test Split
    print('Data Splitting...')
    
    data_len = len(prev_data)
    train_len = int(data_len * 0.8)
    valid_len = int(data_len * 0.1)

    train_index = np.random.choice(data_len, train_len, replace = False) 
    valid_index = np.random.choice(list(set(range(data_len)) - set(train_index)), valid_len, replace = False)
    test_index = set(range(data_len)) - set(train_index) - set(valid_index)

    print(f'train data: {len(train_index)}')
    print(f'valid data: {len(valid_index)}')
    print(f'test data: {len(test_index)}')

    train_data_src = [prev_data[i] for i in train_index]
    train_data_trg = [after_data[i] for i in train_index]
    valid_data_src = [prev_data[i] for i in valid_index]
    valid_data_trg = [after_data[i] for i in valid_index]
    test_data_src = [prev_data[i] for i in test_index]
    test_data_trg = [after_data[i] for i in test_index]

    print('Saving...')
    if not os.path.exists('./preprocessing'):
        os.mkdir('preprocessing')
    hf_data_train = h5py.File(f'./preprocessing/{args.processing_data}_preprocessed_train.h5', 'w')
    hf_data_train.create_dataset(f'train_{args.processing_data}_src', data=train_data_src)
    hf_data_train.create_dataset(f'train_{args.processing_data}_trg', data=train_data_trg)
    hf_data_train.close()

    hf_data_valid = h5py.File(f'./preprocessing/{args.processing_data}_preprocessed_valid.h5', 'w')
    hf_data_valid.create_dataset(f'valid_{args.processing_data}_src', data=valid_data_src)
    hf_data_valid.create_dataset(f'valid_{args.processing_data}_trg', data=valid_data_trg)
    hf_data_valid.close()

    hf_data_test = h5py.File(f'./preprocessing/{args.processing_data}_preprocessed_test.h5', 'w')
    hf_data_test.create_dataset(f'test_{args.processing_data}_src', data=test_data_src)
    hf_data_test.create_dataset(f'test_{args.processing_data}_trg', data=test_data_trg)
    hf_data_test.close()

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='preprocessing traffic data')
    parser.add_argument('--pems_filename', default='./data/pems-bay.h5', type=str, help='path of pems data')
    parser.add_argument('--metr_filename', default='./data/metr-la.h5', type=str, help='path of metr data')
    parser.add_argument('--processing_data', default='pems', type=str, help='which data to process')
    args = parser.parse_args()
    main(args)