
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:31:19 2020

@author: shvarta3
"""

from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import losses

def loss33(y_true, y_pred):

    condition= (K.equal(K.argmax(y_true, axis=1),2) & K.equal(K.argmax(y_pred, axis=1),1))
    then_expression=K.cast([2],dtype='float32')
    else_expression=K.cast([1],dtype='float32')
    weights=tf.keras.backend.switch(condition,then_expression,else_expression)
    
    
    return (weights) * losses.categorical_crossentropy(y_true, y_pred)

