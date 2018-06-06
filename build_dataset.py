"""Split dataset into train/dev/test.  Use 80%/20% split for train/dev.

Train color
raw_data/train_color/170908_072637296_Camera_5.jpg
Train label
raw_data/train_label/170908_072637296_Camera_5_instanceIds.png
Test color
raw_data/test/

170908 = car number
072637296 = image number
5 = left or right camera
"""

import argparse
import random
import os
import numpy as np
import scipy.sparse
from scipy import stats

from PIL import Image
from tqdm import tqdm

tqdm.monitor_interval = 0

ROWSIZE = 564
COLSIZE = 224
CROPSIZE = 1344
SEQ_SIZE = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/users/wdu/raw_data', help="Directory with the raw data")
parser.add_argument('--output_dir', default='/scratch/users/wdu/code/large_data', help="Where to write the new data")
parser.add_argument('--test', default=1, type = int)
parser.add_argument('--train_num', default=0, type = int)

def crop_and_save(images_rand, filename, output_dir):
    for im in images_rand:
        image = Image.open(im)
        w, h = image.size
        image = image.crop((0, h - CROPSIZE, w, h))
        image.save(os.path.join(os.path.join(output_dir, filename.split('/')[-1]), im.split('/')[-1]))
        
def resize_and_save(images_rand, filename, output_dir, rowsize=ROWSIZE, colsize=COLSIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    for im in images_rand:
        image = Image.open(im)
        w, h = image.size
        image = image.crop((0, h - CROPSIZE, w, h))
        # Use bilinear interpolation instead of the default "nearest neighbor" method
        image = image.resize((rowsize, colsize), Image.BILINEAR)
        image.save(os.path.join(os.path.join(output_dir, filename.split('/')[-1]), im.split('/')[-1]))
        
def label_crop_and_save(images_rand, filename, output_dir):
    for im in labels_rand:
        image = Image.open(im)
        w, h = image.size
        image = image.crop((0, h - CROPSIZE, w, h))
        image_num = np.array(image)
        class_lab = image_num / 1000
        class_lab = class_lab.astype(int)
        class_lab = (class_lab - 32) * ((class_lab >= 33) & (class_lab <= 36)) + (class_lab - 33) * ((class_lab >= 38) & (class_lab <= 40))
        instance_lab = image_num % 1000
        instance_lab = instance_lab * (class_lab != 0)
        class_file = (im.split('/')[-1]).split('.png')[0] + '_label_class'
        instance_file = im.split('/')[-1].split('.png')[0] + '_label_instance'
        class_lab_s = scipy.sparse.csr_matrix(class_lab)
        instance_lab_s = scipy.sparse.csr_matrix(instance_lab)
        scipy.sparse.save_npz(os.path.join(os.path.join(output_dir, labelpath.split('/')[-1]), class_file), class_lab_s)
        scipy.sparse.save_npz(os.path.join(os.path.join(output_dir, labelpath.split('/')[-1]), instance_file), instance_lab_s)

def label_resize_and_save(labels_rand, labelpath, output_dir, rowsize=ROWSIZE, colsize=COLSIZE):
    """Resize the label contained in `labelpath` and save it to the `output_dir`"""
    for im in labels_rand:
        image = Image.open(im)
        w, h = image.size
        image = image.crop((0, h - CROPSIZE, w, h))
        image_num = np.array(image)
        old_size = image_num.shape
        old_size_avg = (int(old_size[0] / colsize), int(old_size[1] / rowsize))
        class_lab = image_num / 1000
        class_lab = class_lab.astype(int)
        class_lab = (class_lab - 32) * ((class_lab >= 33) & (class_lab <= 36)) + (class_lab - 33) * ((class_lab >= 38) & (class_lab <= 40))
        instance_lab = image_num % 1000
        instance_lab = instance_lab * (class_lab != 0)
        class_lab_new = np.zeros((colsize, rowsize))
        instance_lab_new = np.zeros((colsize, rowsize))
        for i in range(colsize):
            oldimin = old_size_avg[0]*i
            oldimax = old_size_avg[0]*(i+1)
            class_sub_i = class_lab[oldimin:oldimax,:]
            instance_sub_i = instance_lab[oldimin:oldimax,:]
            for j in range(rowsize):
                oldjmin = old_size_avg[1]*j
                oldjmax = old_size_avg[1]*(j+1)
                class_sub_ij = class_sub_i[:,oldjmin:oldjmax]
                instance_sub_ij = instance_sub_i[:,oldjmin:oldjmax]
                class_lab_new[i, j] = stats.mode(class_sub_ij, axis = None)[0]
                instance_lab_new[i, j] = stats.mode(instance_sub_ij, axis = None)[0]
        class_file = (im.split('/')[-1]).split('.png')[0] + '_label_class'
        instance_file = im.split('/')[-1].split('.png')[0] + '_label_instance'
        class_lab_s = scipy.sparse.csr_matrix(class_lab_new)
        instance_lab_s = scipy.sparse.csr_matrix(instance_lab_new)
        scipy.sparse.save_npz(os.path.join(os.path.join(output_dir, labelpath.split('/')[-1]), class_file), class_lab_s)
        scipy.sparse.save_npz(os.path.join(os.path.join(output_dir, labelpath.split('/')[-1]), instance_file), instance_lab_s)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_color')
    test_data_dir = os.path.join(args.data_dir, 'test')
    train_label_dir = os.path.join(args.data_dir, 'train_label')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, o) for o in filenames if os.path.isdir(os.path.join(train_data_dir, o))]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, o) for o in test_filenames if os.path.isdir(os.path.join(test_data_dir, o))]
    
    # Split the images into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(100)
    filenames.sort()
    random.shuffle(filenames)
    
    
    if args.test == 1:
        filenames = test_filenames
        split = 'test'
    else:
        split = 'train' + str(args.train_num)
        split_tr = 2
        if args.train_num == 22:
            filenames = filenames[(21*split_tr):]
        else:
            filenames = filenames[(split_tr*(args.train_num - 1)):(split_tr*args.train_num)]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
    if not os.path.exists(output_dir_split):
        os.mkdir(output_dir_split)
    else:
        print("Warning: dir {} already exists".format(output_dir_split))

    print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
    for filename in tqdm(filenames):
        output_dir_split_files = os.path.join(output_dir_split, filename.split('/')[-1])
        if not os.path.exists(output_dir_split_files):
            os.mkdir(output_dir_split_files)
        images = [os.path.join(filename, f) for f in sorted(os.listdir(filename)) if f.endswith('.jpg')]
        images_rand = [images[i] for i in range(0, len(images), SEQ_SIZE)]
        resize_and_save(images_rand, filename, output_dir_split, rowsize=ROWSIZE, colsize=COLSIZE)
        # crop_and_save(images_rand, filename, output_dir_split)
        if 'train' in split:
            labelpath = os.path.join(train_label_dir, filename.split('/')[-1])
            labels_rand = [(f.split('/')[-1]).split('.jpg')[0]+'_instanceIds.png' for f in images_rand]
            labels_rand = [os.path.join(labelpath, f) for f in labels_rand]
            label_resize_and_save(labels_rand, labelpath, output_dir_split, rowsize=ROWSIZE, colsize=COLSIZE)
            # label_crop_and_save(labels_rand, labelpath, output_dir_split)
     
    print("Done building dataset")
