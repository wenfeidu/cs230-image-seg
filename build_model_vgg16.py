"""Build convolutional/deconvolutional model in Keras.
"""

import argparse
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import math
import random
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, UpSampling2D, Reshape, multiply, concatenate, Lambda, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.models import Model
import keras.backend as K
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16

# define input and output dimensions
input_row_len = 208
input_col_len = 256
num_classes = 8

# define weights for loss
# w = np.ones(num_classes)
w = np.array([0.01, 0.99/7, 0.99/7, 0.99/7, 0.99/7, 0.99/7, 0.99/7, 0.99/7]) * num_classes

# define tuning parameters
learning_rate = 0.001
k_size = 3
p_size = 4
b_size = 10
epochs = 10 # set to 10-100 for test

# define sample
road = 1 # set to 1, 2, or 3
cam = 5 # set to 5, or 6
sample = 1 # set to 1 for test
namestr = 'road0' + str(road) + '_cam_' + str(cam)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')

def categorical_crossentropy_w(w):
    def loss_calc(y, ypred):
        ypred = K.clip(ypred, K.epsilon(), 1 - K.epsilon())
        loss = -y * K.log(ypred) * K.variable(w)
        loss_sum = K.sum(loss, axis = -1)
        loss_sum = K.mean(loss_sum)
        return loss_sum
    return loss_calc

def image_proc_one(f):
    labi = scipy.sparse.load_npz(f)
    labi = labi.todense()
    labi = np.reshape(labi, (input_row_len * input_col_len, 1))
    labi = to_categorical(labi, num_classes = num_classes)
    imf = f.replace('_instanceIds_label_class.npz','.jpg')
    imi = image.load_img(imf)
    return labi, imi

def image_proc(allfiles, b_size):
    while True:
        ind1 = 0
        while ind1 < len(allfiles):
            ind2 = min(len(allfiles), ind1 + b_size)
            ims = []
            labs = []
            for i in range(ind1, ind2):
                f = allfiles[i]
                labi, imi = image_proc_one(f)
                ims.append(imi)
                labs.append(labi)
            ims = np.stack(ims, axis = 0)
            labs = np.stack(np.array(labs), axis = 0)
            yield (ims, labs)
            ind1 = ind1 + b_size

input = Input(shape=(input_row_len, input_col_len, 3,), dtype='float32')
model_vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input, input_shape=(input_row_len, input_col_len, 3), pooling=None)
model_vgg.trainable=False
up = UpSampling2D(size=(p_size, p_size), data_format='channels_last')(model_vgg.output)
pad = ZeroPadding2D(padding=(1,0))(up)
relu = Activation('relu')(pad)
deconv = Conv2DTranspose(filters=num_classes,kernel_size=(k_size,k_size), padding='same',data_format='channels_last')(relu)
up2 = UpSampling2D(size=(p_size, p_size), data_format='channels_last')(deconv)
relu2 = Activation('relu')(up2)
deconv2 = Conv2DTranspose(filters=num_classes,kernel_size=(k_size,k_size), padding='same',data_format='channels_last')(relu2)
up3 = UpSampling2D(size=(p_size/2, p_size/2), data_format='channels_last')(deconv2)
relu3 = Activation('relu')(up3)
deconv3 = Conv2DTranspose(filters=num_classes,kernel_size=(k_size,k_size), padding='same',data_format='channels_last', name = 'deconv')(relu3)
out = Reshape((input_row_len * input_col_len, num_classes))(deconv3)
out = Activation('softmax')(out)
model = Model(inputs=[input], outputs=[out])
model_inter = Model(inputs=[input], outputs=model.get_layer('deconv').output)
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
model.compile(loss=categorical_crossentropy_w(w), optimizer=Adam(lr=learning_rate), metrics = ['accuracy'])

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train')
    # dev_data_dir = os.path.join(args.data_dir, 'dev')

    # Get the filenames in each directory (train)
    train_dirnames = os.listdir(train_data_dir)
    train_dirnames = [os.path.join(train_data_dir, o) for o in train_dirnames if os.path.isdir(os.path.join(train_data_dir, o))]

    # dev_dirnames = os.listdir(dev_data_dir)
    # dev_dirnames = [os.path.join(dev_data_dir, o) for o in dev_dirnames if os.path.isdir(os.path.join(dev_data_dir, o))]

    allfiles = []
    for dir in train_dirnames:
        if namestr in dir:
            files = os.listdir(dir)
            for f in files:
                if f.endswith('instanceIds_label_class.npz'):
                    if np.random.uniform() < sample:
                        allfiles.append(os.path.join(dir, f))
    
    indtr = list(range(math.floor(len(allfiles) * 0.9)))
    indval = list(range(math.floor(len(allfiles) * 0.9), len(allfiles)))
    ind = np.random.choice(range(len(allfiles)), size = len(allfiles), replace = False)
    allfiles = [allfiles[i] for i in ind]
    allfilestr = [allfiles[i] for i in indtr]
    allfilesval = [allfiles[i] for i in indval]
    modelfit = model.fit_generator(image_proc(allfilestr, b_size), steps_per_epoch = math.floor(len(allfilestr)/b_size) + 1, epochs = epochs, validation_data = image_proc(allfilesval, b_size), validation_steps = math.floor(len(allfilestr)/b_size) + 1)   
    tr_ex = np.array(image_proc_one(allfilestr[0])[1])
    tr_ex = np.reshape(tr_ex, (-1, tr_ex.shape[0], tr_ex.shape[1], tr_ex.shape[2]))
    predtr_ex = model.predict(tr_ex)
    predtr_ex = np.reshape(np.argmax(predtr_ex, axis = 2), (input_row_len, input_col_len))
    print(np.amax(predtr_ex))
    print(np.amin(predtr_ex))
    trnamestr = 'predtr_ex_' + namestr + '.png'
    plt.imsave(trnamestr, predtr_ex, cmap = plt.get_cmap('gray'))
    val_ex = np.array(image_proc_one(allfilesval[0])[1])
    val_ex = np.reshape(val_ex, (-1, val_ex.shape[0], val_ex.shape[1], val_ex.shape[2]))
    predval_ex = model.predict(val_ex)
    predval_ex = np.reshape(np.argmax(predval_ex, axis = 2), (input_row_len, input_col_len))
    valnamestr = 'predval_ex_' + namestr + '.png'
    plt.imsave(valnamestr, predval_ex, cmap = plt.get_cmap('gray'))
    
    plt.figure()
    plt.plot(modelfit.history['acc'])
    plt.plot(modelfit.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    accnamestr = 'acc_' + namestr + '.png'
    plt.savefig(accnamestr)
    
    plt.figure()
    plt.plot(modelfit.history['loss'])
    plt.plot(modelfit.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    lossnamestr = 'loss_' + namestr + '.png'
    plt.savefig(lossnamestr)
    
    wnamestr = 'w_' + namestr + '.h5'
    model.save_weights(wnamestr, overwrite=True)