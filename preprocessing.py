# Import Module
import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(args):
    pems = pd.read_hdf(args.pems_filename) # 03/12 AM02:00 is missing!!

    print('pems shape: ' + str(pems.shape))

    prev_pems = list()
    after_pems = list()
    for col in tqdm(pems.columns):
        for i in range(pems.shape[0] - 24):
            input_ = pems[col].iloc[i:i+12].tolist()
            output_ = pems[col].iloc[i+12:i+24].tolist()
            if 0 not in input_ and 0 not in output_:
                prev_pems.append(input_)
                after_pems.append(output_)

    # Train & Valid & Test Split
    print('Data Splitting...')
    start_time = time.time()

    data_len = len(pems)
    train_len = int(data_len * 0.8)
    valid_len = int(data_len * 0.1)

    train_index = np.random.choice(data_len, train_len, replace = False) 
    valid_index = np.random.choice(list(set(range(data_len)) - set(train_index)), valid_len, replace = False)
    test_index = set(range(data_len)) - set(train_index) - set(valid_index)

    print(f'train data: {len(train_index)}')
    print(f'valid data: {len(valid_index)}')
    print(f'test data: {len(test_index)}')

    train_pems_src = [prev_pems[i] for i in train_index]
    train_pems_trg = [after_pems[i] for i in train_index]
    valid_pems_src = [prev_pems[i] for i in valid_index]
    valid_pems_trg = [after_pems[i] for i in valid_index]
    test_pems_src = [prev_pems[i] for i in test_index]
    test_pems_trg = [after_pems[i] for i in test_index]

    print('Saving...')
    if not os.path.exists('./preprocessing'):
        os.mkdir('preprocessing')
    hf_pems = h5py.File('./preprocessing/pems_preprocessed.h5', 'w')
    hf_pems.create_dataset('previous', data=prev_pems)
    hf_pems.create_dataset('after', data=after_pems)
    hf_pems.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='preprocessing traffic data')
    parser.add_argument('--pems_filename', default='./data/pems-bay.h5', type=str, help='path of pems data')
    parser.add_argument('--metr_filename', default='./data/metr-la.h5', type=str, help='path of metr data')
    args = parser.parse_args()
    main(args)