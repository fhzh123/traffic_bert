# Import Module
import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(args):
    pems = pd.read_hdf(args.pems_filename) # 03/12 AM02:00 is missing!!
    metr = pd.read_hdf(args.metr_filename)

    print('pems shape: ' + str(pems.shape))
    print('metr shape: ' + str(metr.shape))

    prev_pems = list()
    after_pems = list()
    for col in tqdm(pems.columns):
        for i in range(pems.shape[0] - 24):
            input_ = pems[col].iloc[i:i+12]
            output_ = pems[col].iloc[i+12:i+24]
            prev_pems.append(input_.tolist())
            after_pems.append(output_.tolist())

    prev_metr = list()
    after_metr = list()
    for col in tqdm(metr.columns):
        for i in range(metr.shape[0] - 24):
            input_ = metr[col].iloc[i:i+12]
            output_ = metr[col].iloc[i+12:i+24]
            prev_metr.append(input_.tolist())
            after_metr.append(output_.tolist())

    print('Saving...')
    if not os.path.exists('./preprocessing'):
        os.mkdir('preprocessing')
    hf_pems = h5py.File('./preprocessing/pems_preprocessed.h5', 'w')
    hf_pems.create_dataset('previous', data=prev_pems)
    hf_pems.create_dataset('after', data=after_pems)
    hf_pems.close()

    hf_metr = h5py.File('./preprocessing/metr_preprocessed.h5', 'w')
    hf_metr.create_dataset('previous', data=prev_metr)
    hf_metr.create_dataset('after', data=after_metr)
    hf_metr.close()
    print('Done!')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='preprocessing traffic data')
    parser.add_argument('--pems_filename', default='./data/pems-bay.h5', type=str, help='path of pems data')
    parser.add_argument('--metr_filename', default='./data/metr-la.h5', type=str, help='path of metr data')
    args = parser.parse_args()
    main(args)