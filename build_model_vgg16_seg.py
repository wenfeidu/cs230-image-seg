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
from keras import regularizers

from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='large_data')
parser.add_argument('--output_dir', default='output')
parser.add_argument('--lam1', default=0.0)
parser.add_argument('--lam2', default=0.0)

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
    return tf.where(K.equal(x[0], x[1]), x[2], tf.zeros_like(x[2]))

def cropping(x):
    return x[:,pad_row:(input_row_len + pad_row),pad_col:(input_col_len + pad_col),:]
    
def model_calc(lam1, lam2):
    input = Input(shape=(input_row_len, input_col_len, 3,), dtype='float32')
    input_pad = ZeroPadding2D(padding=(pad_row,pad_col))(input)
    model_vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_pad, input_shape=(input_row_len + 2*pad_row, input_col_len + 2*pad_col, 3), pooling=None)
    model_vgg.trainable=False
    up5 = UpSampling2D(size=(2, 2), data_format='channels_last')(model_vgg.output)
    orig5 = UpSampling2D(size=(2, 2), data_format='channels_last')(model_vgg.get_layer('block5_pool').output)
    unpool5 = Lambda(unpooling)([orig5, model_vgg.get_layer('block5_conv3').output, up5])
    conv5 = Conv2D(filters=512, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(unpool5)
    conv5 = Conv2D(filters=512, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv5)
    conv5 = Conv2D(filters=512, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv5)
    up4 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)
    orig4 = UpSampling2D(size=(2, 2), data_format='channels_last')(model_vgg.get_layer('block4_pool').output)
    unpool4 = Lambda(unpooling)([orig4, model_vgg.get_layer('block4_conv3').output, up4])
    conv4 = Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(unpool4)
    conv4 = Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv4)
    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv4)
    orig3 = UpSampling2D(size=(2, 2), data_format='channels_last')(model_vgg.get_layer('block3_pool').output)
    unpool3 = Lambda(unpooling)([orig3, model_vgg.get_layer('block3_conv3').output, up3])
    conv3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(unpool3)
    conv3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv3)
    up2 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv3)
    orig2 = UpSampling2D(size=(2, 2), data_format='channels_last')(model_vgg.get_layer('block2_pool').output)
    unpool2 = Lambda(unpooling)([orig2, model_vgg.get_layer('block2_conv2').output, up2])
    conv2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(unpool2)
    conv2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(conv2)
    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv2)
    orig1 = UpSampling2D(size=(2, 2), data_format='channels_last')(model_vgg.get_layer('block1_pool').output)
    unpool1 = Lambda(unpooling)([orig1, model_vgg.get_layer('block1_conv2').output, up1])
    convout = Conv2D(filters=num_classes, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(unpool1)
    convout = Conv2D(filters=num_classes, kernel_size=(3,3), activation='relu',padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(convout)
    convout = Conv2D(filters=num_classes, kernel_size=(3,3), padding='same',data_format='channels_last', kernel_regularizer=regularizers.l2(lam2), activity_regularizer=regularizers.l1(lam1))(convout)
    convout = Lambda(cropping)(convout)
    out = Reshape((input_row_len * input_col_len, num_classes))(convout)
    out = Activation('softmax')(out)
    model = Model(inputs=[input], outputs=[out])
    model.summary()
    model.compile(loss=categorical_crossentropy_w(w), optimizer=Adam(lr=learning_rate), metrics = [iou_hard, iou_soft, true_pos, true_neg, false_pos, false_neg])
    return model
    
if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train')
    dev_data_dir = os.path.join(args.data_dir, 'trainval')

    # Get the filenames in each directory (train)
    train_dirnames = os.listdir(train_data_dir)
    train_dirnames = [os.path.join(train_data_dir, o) for o in train_dirnames if os.path.isdir(os.path.join(train_data_dir, o))]

    dev_dirnames = os.listdir(dev_data_dir)
    dev_dirnames = [os.path.join(dev_data_dir, o) for o in dev_dirnames if os.path.isdir(os.path.join(dev_data_dir, o))]

    allfilestr = []
    for dir in train_dirnames:
        if namestr in dir:
            files = os.listdir(dir)
            for f in files:
                if f.endswith('instanceIds_label_class.npz'):
                    if np.random.uniform() < sample:
                        allfilestr.append(os.path.join(dir, f))
    
    allfilesval = []
    for dir in dev_dirnames:
        if namesval in dir:
            files = os.listdir(dir)
            for f in files:
                if f.endswith('instanceIds_label_class.npz'):
                    if np.random.uniform() < sample:
                        allfilesval.append(os.path.join(dir, f))
    
    print(len(allfilestr))
    posfilestr = []
    negfilestr = []
    for f in allfilestr:
        labi = scipy.sparse.load_npz(f)
        labi = labi.todense()
        if (np.sum(labi == 0) / labi.size) < cutoff:
            posfilestr.append(f)
        else:
            negfilestr.append(f)
    negfilestr = random.sample(negfilestr, len(posfilestr))
    newfilestr = posfilestr + negfilestr
    random.seed(100)
    newfilestr.sort()
    random.shuffle(newfilestr)
    print(len(newfilestr))
    
    model = model_calc(args.lam1, args.lam2)
    modelfit = model.fit_generator(image_proc(newfilestr, b_size), steps_per_epoch = math.floor(len(newfilestr)/b_size) + 1, epochs = epochs, validation_data = image_proc(allfilesval, b_size), validation_steps = math.floor(len(allfilesval)/b_size) + 1)   
    
    print(newfilestr[extr])
    tr_ex = np.array(image_proc_one(newfilestr[extr])[1])
    tr_ex = np.reshape(tr_ex, (-1, tr_ex.shape[0], tr_ex.shape[1], tr_ex.shape[2]))
    tr_ex_lab = scipy.sparse.load_npz(newfilestr[extr]).todense()
    predtr_ex = model.predict(tr_ex)
    predtr_ex = np.reshape(np.argmax(predtr_ex, axis = 2), (input_row_len, input_col_len))
    trnamestr = os.path.join(args.output_dir, 'predtr_ex_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namestr + '.png')
    plt.imsave(trnamestr, predtr_ex, cmap = plt.get_cmap('gray'))
    trnamestr_lab = os.path.join(args.output_dir, 'labtr_ex_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namestr + '.png')
    plt.imsave(trnamestr_lab, tr_ex_lab, cmap = plt.get_cmap('gray'))
    
    print(allfilesval[exval])
    val_ex = np.array(image_proc_one(allfilesval[exval])[1])
    val_ex = np.reshape(val_ex, (-1, val_ex.shape[0], val_ex.shape[1], val_ex.shape[2]))
    val_ex_lab = scipy.sparse.load_npz(allfilesval[exval]).todense()
    predval_ex = model.predict(val_ex)
    predval_ex = np.reshape(np.argmax(predval_ex, axis = 2), (input_row_len, input_col_len))
    valnamestr = os.path.join(args.output_dir, 'predval_ex_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namesval + '.png')
    plt.imsave(valnamestr, predval_ex, cmap = plt.get_cmap('gray'))
    valnamestr_lab = os.path.join(args.output_dir, 'labval_ex_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namesval + '.png')
    plt.imsave(valnamestr_lab, val_ex_lab, cmap = plt.get_cmap('gray'))
    
    plt.figure()
    plt.plot(modelfit.history['iou_soft'])
    plt.plot(modelfit.history['val_iou_soft'])
    plt.plot(modelfit.history['iou_hard'])
    plt.plot(modelfit.history['val_iou_hard'])
    plt.title('IOU')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['iou_soft', 'val_iou_soft', 'iou_hard', 'val_iou_hard'], loc='upper left')
    iounamestr = os.path.join(args.output_dir, 'iou_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namestr + '.png')
    plt.savefig(iounamestr)
    
    plt.figure()
    plt.plot(modelfit.history['true_neg'])
    plt.plot(modelfit.history['true_pos'])
    plt.plot(modelfit.history['false_neg'])
    plt.plot(modelfit.history['false_pos'])
    plt.title('Misclassifications')
    plt.ylabel('number')
    plt.xlabel('epoch')
    plt.legend(['val_true_neg', 'val_true_pos', 'val_false_neg', 'val_false_pos'], loc='upper left')
    iounamestr = os.path.join(args.output_dir, 'mis_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namestr + '.png')
    plt.savefig(iounamestr)
    
    plt.figure()
    plt.plot(modelfit.history['loss'])
    plt.plot(modelfit.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    lossnamestr = os.path.join(args.output_dir, 'loss_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namestr + '.png')
    plt.savefig(lossnamestr)
    
    wnamestr = os.path.join(args.output_dir, 'w_seg_reg' + str(args.lam1) + '_' + str(args.lam2) + '_' + namestr + '.h5')
    model.save(wnamestr, overwrite=True)
