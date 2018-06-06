import keras.backend as K
import tensorflow as tf
import numpy as np

# define input and output dimensions
input_row_len = 224
input_col_len = 564
# input_row_len = 208
# input_col_len = 256
num_classes = 8
pad_row = 0
pad_col = 6
# pad_row = 8
# pad_col = 0

# define tuning parameters
learning_rate = 0.001
k_size = 3
b_size = 10
epochs = 1 # set to 10-100 for test

# define training sample
road = 1 # set to 1, 2, or 3
cam = 5 # set to 5, or 6
sample = 0.02 # set to 1 for test
# namestr = 'road0' + str(road) + '_cam_' + str(cam)
namestr = ''
extr = 0

# define validation sample
road = 1 # set to 1, 2, or 3
cam = 5 # set to 5, or 6
sample = 0.02 # set to 1 for test
# namesval = 'road0' + str(road) + '_cam_' + str(cam)
namesval = ''
exval = 0

# define weights for loss
# w = np.ones(num_classes)
cutoff = 0.9
w = np.array([0.01, 0.99/7, 0.99/7, 0.99/7, 0.99/7, 0.99/7, 0.99/7, 0.99/7]) * num_classes

def categorical_crossentropy_w(w):
    def loss_calc(y, ypred):
        ypred = K.clip(ypred, K.epsilon(), 1 - K.epsilon())
        loss = -y * K.log(ypred) * K.variable(w)
        loss_sum = K.sum(loss, axis = -1)
        loss_sum = K.mean(loss_sum)
        return loss_sum
    return loss_calc

# define true and false rate metrics
def true_pos(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis = -1)
    y_true = K.argmax(y_true, axis = -1)
    testsum = tf.constant(0, tf.float32)
    for i in range(1, num_classes):
        ti = tf.constant(i, shape = [input_row_len * input_col_len], dtype = tf.int64)
        testsum = testsum + K.mean(tf.cast(tf.equal(y_true, ti), tf.float32) * tf.cast(tf.equal(y_pred, ti), tf.float32))
    return testsum

def false_pos(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis = -1)
    y_true = K.argmax(y_true, axis = -1)
    t0 = tf.constant(0, shape = [input_row_len * input_col_len], dtype = tf.int64)
    testsum = tf.constant(0, tf.float32)
    for i in range(1, num_classes):
        ti = tf.constant(i, shape = [input_row_len * input_col_len], dtype = tf.int64)
        testsum = testsum + K.mean(tf.cast(tf.equal(y_true, t0), tf.float32) * tf.cast(tf.equal(y_pred, ti), tf.float32))
    return testsum

def true_neg(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis = -1)
    y_true = K.argmax(y_true, axis = -1)
    t0 = tf.constant(0, shape = [input_row_len * input_col_len], dtype = tf.int64)
    return K.mean(tf.cast(tf.equal(y_true, t0), tf.float32) * tf.cast(tf.equal(y_pred, t0), tf.float32))

def false_neg(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis = -1)
    y_true = K.argmax(y_true, axis = -1)
    t0 = tf.constant(0, shape = [input_row_len * input_col_len], dtype = tf.int64)
    testsum = tf.constant(0, tf.float32)
    for i in range(1, num_classes):
        ti = tf.constant(i, shape = [input_row_len * input_col_len], dtype = tf.int64)
        testsum = testsum + K.mean(tf.cast(tf.equal(y_true, ti), tf.float32) * tf.cast(tf.equal(y_pred, t0), tf.float32))
    return testsum
    
# define iou metrics
def iou_soft(y_true, y_pred):
    y_true = K.argmax(y_true, axis = -1)
    testsum = tf.constant(0.0)
    numsum = tf.constant(0.0)
    t0 = tf.constant(0.0)
    for i in range(1, num_classes):
        ti = tf.constant(i, shape = [input_row_len * input_col_len], dtype = tf.int64)
        ii = tf.reduce_sum(tf.cast(tf.equal(y_true, ti), tf.float32) * y_pred[:,:,i], 1)
        ui = tf.reduce_sum(tf.cast(tf.equal(y_true, ti), tf.float32) + y_pred[:,:,i], 1) - ii
        test = ii/ui
        testsum = testsum + tf.where(tf.logical_not(tf.is_nan(test)), test, tf.zeros_like(test))
        numsum = numsum + tf.where(tf.logical_not(tf.is_nan(test)), tf.ones_like(test), tf.zeros_like(test))
    testmean = testsum / numsum
    testmean = tf.where(tf.logical_not(tf.is_nan(testmean)), testmean, tf.zeros_like(testmean))
    return K.mean(testmean)
    
def iou_hard(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis = -1)
    y_true = K.argmax(y_true, axis = -1)
    testsum = tf.constant(0.0)
    numsum = tf.constant(0.0)
    t0 = tf.constant(0.0)
    for i in range(1, num_classes):
        ti = tf.constant(i, shape = [input_row_len * input_col_len], dtype = tf.int64)
        ii = tf.reduce_sum(tf.cast(tf.equal(y_true, ti), tf.float32) * tf.cast(tf.equal(y_pred, ti), tf.float32), 1)
        ui = tf.reduce_sum(tf.cast(tf.equal(y_true, ti), tf.float32) + tf.cast(tf.equal(y_pred, ti), tf.float32), 1) - ii
        test = ii/ui
        testsum = testsum + tf.where(tf.logical_not(tf.is_nan(test)), test, tf.zeros_like(test))
        numsum = numsum + tf.where(tf.logical_not(tf.is_nan(test)), tf.ones_like(test), tf.zeros_like(test))
    testmean = testsum / numsum
    testmean = tf.where(tf.logical_not(tf.is_nan(testmean)), testmean, tf.zeros_like(testmean))
    return K.mean(testmean)
