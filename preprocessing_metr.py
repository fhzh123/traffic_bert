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

    metr = pd.read_hdf(args.metr_filename) # 03/12 AM02:00 is missing!!

    print('metr shape: ' + str(metr.shape))

    prev_metr = list()
    after_metr = list()
    for col in tqdm(metr.columns):
        for i in range(metr.shape[0] - 24):
            input_ = metr[col].iloc[i:i+12].tolist()
            output_ = metr[col].iloc[i+12:i+24].tolist()
            if 0 not in input_ and 0 not in output_:
                prev_metr.append(input_)
                after_metr.append(output_)

    # Train & Valid & Test Split
    print('Data Splitting...')
    
    data_len = len(prev_metr)
    train_len = int(data_len * 0.8)
    valid_len = int(data_len * 0.1)

    train_index = np.random.choice(data_len, train_len, replace = False) 
    valid_index = np.random.choice(list(set(range(data_len)) - set(train_index)), valid_len, replace = False)
    test_index = set(range(data_len)) - set(train_index) - set(valid_index)

    print(f'train data: {len(train_index)}')
    print(f'valid data: {len(valid_index)}')
    print(f'test data: {len(test_index)}')

    train_metr_src = [prev_metr[i] for i in train_index]
    train_metr_trg = [after_metr[i] for i in train_index]
    valid_metr_src = [prev_metr[i] for i in valid_index]
    valid_metr_trg = [after_metr[i] for i in valid_index]
    test_metr_src = [prev_metr[i] for i in test_index]
    test_metr_trg = [after_metr[i] for i in test_index]

    print('Saving...')
    if not os.path.exists('./preprocessing'):
        os.mkdir('preprocessing')
    hf_metr_train = h5py.File('./preprocessing/metr_preprocessed_train.h5', 'w')
    hf_metr_train.create_dataset('train_metr_src', data=train_metr_src)
    hf_metr_train.create_dataset('train_metr_trg', data=train_metr_trg)
    hf_metr_train.close()

    hf_metr_valid = h5py.File('./preprocessing/metr_preprocessed_valid.h5', 'w')
    hf_metr_valid.create_dataset('valid_metr_src', data=valid_metr_src)
    hf_metr_valid.create_dataset('valid_metr_trg', data=valid_metr_trg)
    hf_metr_valid.close()

    hf_metr_test = h5py.File('./preprocessing/metr_preprocessed_test.h5', 'w')
    hf_metr_test.create_dataset('test_metr_src', data=test_metr_src)
    hf_metr_test.create_dataset('test_metr_trg', data=test_metr_trg)
    hf_metr_test.close()

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='preprocessing traffic data')
    parser.add_argument('--pems_filename', default='./data/pems-bay.h5', type=str, help='path of pems data')
    parser.add_argument('--metr_filename', default='./data/metr-la.h5', type=str, help='path of metr data')
    args = parser.parse_args()
    main(args)