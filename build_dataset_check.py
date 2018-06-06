import argparse
import os
import math
import random
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='large_data')
parser.add_argument('--output_dir', default='output')

input_row_len = 224
input_col_len = 564
num_classes = 8
sample = 1

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train')

    # Get the filenames in each directory (train)
    train_dirnames = os.listdir(train_data_dir)
    train_dirnames = [os.path.join(train_data_dir, o) for o in train_dirnames if os.path.isdir(os.path.join(train_data_dir, o))]

    allfilestr = []
    for dir in train_dirnames:
        files = os.listdir(dir)
        for f in files:
            if f.endswith('instanceIds_label_class.npz'):
                if np.random.uniform() < sample:
                    allfilestr.append(os.path.join(dir, f))
                        
    counts = np.zeros((len(allfilestr), num_classes))
    for j,f in enumerate(allfilestr):
        labi = scipy.sparse.load_npz(f)
        labi = labi.todense()
        for i in range(num_classes):
            counts[j,i] = (np.sum(labi == i) / labi.size)
    
    for i in range(num_classes):
        print(np.median(counts[:,i]))
        plt.figure()
        plt.hist(counts[:,i])
        namestr = os.path.join(args.output_dir, 'counts' + str(i) + '.png')
        plt.savefig(namestr)