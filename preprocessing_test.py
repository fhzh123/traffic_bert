# Import Module
import os
import h5py
import time
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from random import shuffle

def main(args):
    start_time = time.time()

    data_dict = {
        'pems_l': pd.read_csv(args.pems_l_filename, index_col='date'),
        'pems_bay': pd.read_hdf(args.pems_bay_filename), # 03/12 AM02:00 is missing!!
        'metr': pd.read_hdf(args.metr_filename)
    }

    print(f'pems_l shape: ' + str(data_dict['pems_l'].shape))
    print(f'pems_bay shape: ' + str(data_dict['pems_bay'].shape))
    print(f'metr shape: ' + str(data_dict['metr'].shape))

    prev_data = list()
    after_data = list()
    weekday_data = list()

    dat = 'pems_bay'

    data = data_dict[dat]

    if dat == 'pems_l':
        print(data.columns)
        data.index = pd.to_datetime(data.index)
    
    col_list = list(data.columns)
    shuffle(col_list)
    
    for col in tqdm(col_list[:10]):
        if dat == 'pems_l':
            for i in range(data.shape[0]):
                input_ = data[col].iloc[i:i+12].tolist()
                output_ = data[col].iloc[i+12:i+24].tolist()
                if 0 not in input_ and 0 not in output_:
                    prev_data.append(input_)
                    after_data.append(output_)
                    weekday_data.append(data.index[i].weekday())
        else:
            for i in range(data.shape[0]):
                input_ = data[col].iloc[i:i+12].tolist()
                output_ = data[col].iloc[i+12:i+24].tolist()
                if 0 not in input_ and 0 not in output_:
                    prev_data.append(input_)
                    after_data.append(output_)
                    weekday_data.append(data.index[i].weekday())
    print('Done!')

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
    train_data_src_week = [weekday_data[i] for i in train_index]
    train_data_trg = [after_data[i] for i in train_index]
    valid_data_src = [prev_data[i] for i in valid_index]
    valid_data_src_week = [weekday_data[i] for i in valid_index]
    valid_data_trg = [after_data[i] for i in valid_index]
    test_data_src = [prev_data[i] for i in test_index]
    test_data_src_week = [weekday_data[i] for i in test_index]
    test_data_trg = [after_data[i] for i in test_index]

    print('Saving...')
    if not os.path.exists('./preprocessing/total2'):
        os.mkdir('preprocessing/total2')

    hf_data_test = h5py.File(f'./preprocessing/total2/preprocessed_test_pems_bay.h5', 'w')
    hf_data_test.create_dataset(f'test_src', data=test_data_src)
    hf_data_test.create_dataset(f'test_src_week', data=test_data_src_week)
    hf_data_test.create_dataset(f'test_trg', data=test_data_trg)
    hf_data_test.close()

    spend_time = round((time.time() - start_time) / 60, 4)
    print(f'Done...! / {spend_time}min spend...!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='preprocessing traffic data')
    parser.add_argument('--pems_bay_filename', default='./data/pems-bay.h5', type=str, help='path of pems bay data')
    parser.add_argument('--metr_filename', default='./data/metr-la.h5', type=str, help='path of metr data')
    parser.add_argument('--pems_l_filename', default='./data/pems_l2.csv', type=str, help='path of pems l data')
    args = parser.parse_args()
    main(args)