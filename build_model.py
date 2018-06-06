"""Build convolutional/deconvolutional model in Keras.
"""

import argparse
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import math
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, UpSampling2D, Reshape, multiply, concatenate, Lambda
from keras.optimizers import Adam, SGD
from keras.models import Model
import keras.backend as K
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

# define input and output dimensions
input_row_len = 208
input_col_len = 256
num_classes = 8
# w = np.ones(num_classes)
w = np.array([.01/8, .99/(7*8), .99/(7*8), .99/(7*8), .99/(7*8), .99/(7*8), .99/(7*8), .99/(7*8)]) * num_classes

# define tuning parameters
learning_rate = 0.001
k_size = 3
p_size = 4
b_size = 10
epochs = 10

# calculated dimensions for intermediate layer
new_row_len = int(input_row_len/p_size)
new_col_len = int(input_col_len/p_size)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='code\data')

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
            
def unpooling(x):
    return tf.where(K.equal(x[:,:,:,:,0], x[:,:,:,:,1]), x[:,:,:,:,2], tf.zeros_like(x[:,:,:,:,2]))

input = Input(shape=(input_row_len, input_col_len, 3,), dtype='float32')
conv1 = Conv2D(filters=64,kernel_size=(k_size, k_size), padding='same',data_format='channels_last')(input)
act1 = Activation('relu')(conv1)
max1 = MaxPooling2D(pool_size=(p_size, p_size), padding='same',data_format='channels_last')(act1)
up1 = UpSampling2D(size=(p_size, p_size), data_format='channels_last')(max1)
act1_r = Reshape((input_row_len, input_col_len, 64, 1))(act1)
up1_r = Reshape((input_row_len, input_col_len, 64, 1))(up1)
conv2 = Conv2D(filters=128,input_shape=(input_row_len, input_col_len, 3), kernel_size=(k_size, k_size), padding='same',data_format='channels_last')(max1)
act2 = Activation('relu')(conv2)
max2 = MaxPooling2D(pool_size=(p_size, p_size), padding='same',data_format='channels_last')(act2)
up2 = UpSampling2D(size=(p_size, p_size), data_format='channels_last')(max2)
act2_r = Reshape((new_row_len, new_col_len, 128, 1))(act2)
up2_r = Reshape((new_row_len, new_col_len, 128, 1))(up2)
# conv3 = Conv2D(filters=256,input_shape=(input_row_len, input_col_len, 3), kernel_size=(k_size, k_size),data_format='channels_last')(max2)
# act3 = Activation('relu')(conv2)
# max3 = MaxPooling2D(data_format='channels_last')(act3)
# conv4 = Conv2D(filters=512,input_shape=(input_row_len, input_col_len, 3), kernel_size=(k_size, k_size),data_format='channels_last')(max3)
# act4 = Activation('relu')(conv4)
# max4 = MaxPooling2D(data_format='channels_last')(act4)
de_conc2 = concatenate([act2_r, up2_r, up2_r])
de_up2 = Lambda(unpooling)(de_conc2)
de_relu2 = Activation('relu')(de_up2)
deconv2 = Conv2DTranspose(filters=64,kernel_size=(k_size,k_size), padding='same',data_format='channels_last')(de_relu2)
de_up1 = UpSampling2D(size=(p_size, p_size), data_format='channels_last')(deconv2)
de_up1_r = Reshape((input_row_len, input_col_len, 64, 1))(de_up1)
de_conc1 = concatenate([act1_r, up1_r, de_up1_r])
de_up1 = Lambda(unpooling)(de_conc1)
de_relu1 = Activation('relu')(de_up1)
deconv1 = Conv2DTranspose(filters=num_classes,kernel_size=(k_size,k_size), padding='same',data_format='channels_last', name = 'deconv')(de_relu1)
out = Reshape((input_row_len * input_col_len, num_classes))(deconv1)
out = Activation('softmax')(out)
model = Model(inputs=[input], outputs=[out])
model_inter = Model(inputs=[input], outputs=model.get_layer('deconv').output)
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
model.compile(loss=categorical_crossentropy_w(w), optimizer=Adam(lr=learning_rate))

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_small')
    # dev_data_dir = os.path.join(args.data_dir, 'dev')

    # Get the filenames in each directory (train)
    train_dirnames = os.listdir(train_data_dir)
    train_dirnames = [os.path.join(train_data_dir, o) for o in train_dirnames if os.path.isdir(os.path.join(train_data_dir, o))]

    # dev_dirnames = os.listdir(dev_data_dir)
    # dev_dirnames = [os.path.join(dev_data_dir, o) for o in dev_dirnames if os.path.isdir(os.path.join(dev_data_dir, o))]

    allfiles = []
    for dir in train_dirnames:
        # if 'road01_cam_5_video_10_image_list_train' in dir:
        files = os.listdir(dir)
        for f in files:
            if f.endswith('instanceIds_label_class.npz'):
                allfiles.append(os.path.join(dir, f))
    
    indtr = list(range(math.floor(len(allfiles) * 0.9)))
    indval = list(range(math.floor(len(allfiles) * 0.9), len(allfiles)))
    ind = np.random.choice(range(len(allfiles)), size = len(allfiles), replace = False)
    allfiles = [allfiles[i] for i in ind]
    allfilestr = [allfiles[i] for i in indtr]
    allfilesval = [allfiles[i] for i in indval]
    model.fit_generator(image_proc(allfilestr, b_size), steps_per_epoch = math.floor(len(allfilestr)/b_size) + 1, epochs = epochs, validation_data = image_proc(allfilesval, b_size), validation_steps = math.floor(len(allfilestr)/b_size) + 1)
    tr_ex = np.array(image_proc_one(allfilestr[0])[1])
    tr_ex = np.reshape(tr_ex, (-1, tr_ex.shape[0], tr_ex.shape[1], tr_ex.shape[2]))
    predtr_ex = model.predict(tr_ex)
    predtr_ex = np.reshape(np.argmax(predtr_ex, axis = 2), (input_row_len, input_col_len))
    print(np.amax(predtr_ex))
    print(np.amin(predtr_ex))
    plt.imsave('code\predtr_ex.png', predtr_ex)
    val_ex = np.array(image_proc_one(allfilesval[0])[1])
    val_ex = np.reshape(val_ex, (-1, val_ex.shape[0], val_ex.shape[1], val_ex.shape[2]))
    predval_ex = model.predict(val_ex)
    predval_ex = np.reshape(np.argmax(predval_ex, axis = 2), (input_row_len, input_col_len))
    plt.imsave('code\predval_ex.png', predval_ex)