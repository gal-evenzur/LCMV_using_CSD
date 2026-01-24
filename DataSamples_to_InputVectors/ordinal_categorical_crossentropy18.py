
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:31:19 2020

@author: shvarta3
"""
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import losses

def loss1(y_true, y_pred):

    condition = (K.equal(K.argmax(y_pred, axis=1),0) | K.equal(K.argmax(y_pred, axis=1),19))
    then_expression=K.cast([5],dtype='float32')
    else_expression = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 2),
                  dtype='float32')
    weights=tf.keras.backend.switch(condition,then_expression,else_expression)
    
#    dist = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1)),dtype='float32')
#    condition_dist1 = K.equal(dist,1)
#    then_expression=K.cast([2],dtype='float32')
#    else_expression = weights
#    weights=tf.keras.backend.switch(condition_dist1,then_expression,else_expression)
    
    condition= (K.equal(K.argmax(y_true, axis=1),0) | K.equal(K.argmax(y_true, axis=1),19))
    then_expression=K.cast([-1],dtype='float32')
    else_expression = weights
    weights=tf.keras.backend.switch(condition,then_expression,else_expression)
    
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

