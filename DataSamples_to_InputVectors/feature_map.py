#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 03:11:31 2021

@author: shvarta3
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
import shap
################################## load model ################

plt.close("all")

forlder_to_work = 'C:/project/'
model_two_channel = load_model(forlder_to_work+'models/model_GEVD_18_separate_22_04.h5',compile=False)
model3=load_model(forlder_to_work+'models/model_conv2d_02_07.h5',compile=False)
#model18=load_model(forlder_to_work+'models/model2_conv2d_02_07.h5',compile=False)
model3_csd_channel=load_model(forlder_to_work+'models/model_only_csd_02_07.h5',compile=False)


################################## load data ##############

x_test_class0 = np.load(r'C:\project\val_data_set\x_test_class0.npy') 
x_test_class1 = np.load(r'C:\project\val_data_set\x_test_class1.npy') 
x_test_class2 = np.load(r'C:\project\val_data_set\x_test_class2.npy') 

################################## new models after first cnn ##############

#model3_first_cnn = Model(inputs=model3.inputs, outputs=model3.layers[1].output)
#model18_first_cnn = Model(inputs=model18.inputs, outputs=model18.layers[1].output)
#model3_first_cnn_only_csd = Model(inputs=model3_csd_channel.inputs, outputs=model3_csd_channel.layers[1].output)

#for layer in model3_first_cnn.layers:
#    print(layer.output_shape)

############################## shap ############################################
X = np.concatenate((x_test_class0[np.random.choice(range(len(x_test_class0)), size=300)],x_test_class1[np.random.choice(range(len(x_test_class1)), size=300)],x_test_class2[np.random.choice(range(len(x_test_class2)), size=300)]))
to_explain = np.concatenate((x_test_class0[np.random.choice(range(len(x_test_class0)), size=30)],x_test_class1[np.random.choice(range(len(x_test_class1)), size=30)],x_test_class2[np.random.choice(range(len(x_test_class2)), size=30)]))

# explain how the input to the 7th layer of the model explains the top two classes
def map2layer(model,x,layer):
    feed_dict = dict(zip([model.layers[0].input],[x]))
    return tf.compat.v1.keras.backend.get_session().run(model.layers[layer].input, feed_dict)


e_two_channel = shap.GradientExplainer(
    (model3.layers[1].input, model3.layers[-1].output),
    map2layer(model3,X, 1),
    local_smoothing=0 # std dev of smoothing noise
)
shap_values_two_channel,indexes_e_two_channel = e_two_channel.shap_values(map2layer(model3,to_explain,1), ranked_outputs=1)


e_only_csd = shap.GradientExplainer(
    (model3_csd_channel.layers[1].input, model3_csd_channel.layers[-1].output),
    map2layer(model3_csd_channel,X, 1),
    local_smoothing=0 # std dev of smoothing noise
)
shap_values_only_csd,indexes_only_csd = e_only_csd.shap_values(map2layer(model3_csd_channel,to_explain,1), ranked_outputs=1)

#get the names for the classes
#index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap.image_plot(shap_values_two_channel,to_explain)
shap.image_plot(shap_values_only_csd,to_explain)
################################## predict by category ######################

## get feature map for first hidden layer
#feature_maps_model3_first_cnn_class_0 = model3_first_cnn.predict(x_test_class0)
#feature_maps_model3_first_cnn_class_1 = model3_first_cnn.predict(x_test_class1)
#feature_maps_model3_first_cnn_class_2 = model3_first_cnn.predict(x_test_class2)
#feature_maps_model18_first_cnn = model18_first_cnn.predict(x_test_class1)
#
#feature_maps_model3_first_cnn_only_csd_class_0 = model3_first_cnn_only_csd.predict(x_test_class0)
#feature_maps_model3_first_only_csd_cnn_class_1 = model3_first_cnn_only_csd.predict(x_test_class1)
#feature_maps_model3_first_only_csd_cnn_class_2 = model3_first_cnn_only_csd.predict(x_test_class2)


















