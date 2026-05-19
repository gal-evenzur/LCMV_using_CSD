# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:05:40 2019

@author: shvarta3
"""

from __future__ import print_function
import os

# 1. Hide the TensorFlow C++ Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. Your existing GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import gmtime, strftime
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
import ordinal_categorical_crossentropy18 as OCC
import ordinal_categorical_crossentropy3 as OCC3
from keras.constraints import max_norm

# ---------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------
pydir = os.path.dirname(os.path.realpath(__file__))
workspace_dir = os.path.dirname(pydir)
dataset_dir = os.path.join(workspace_dir, 'data', 'dataset')

# Training files
train_data_prefix = os.path.join(dataset_dir, 'train/feature_vector_')
train_speaker_labels_prefix = os.path.join(dataset_dir, 'train/label_')
train_angle_labels_prefix = os.path.join(dataset_dir, 'train/label2_')
train_idx_file = os.path.join(dataset_dir, 'train/idx.npy')

# Validation files
val_data_prefix = os.path.join(dataset_dir, 'val/feature_vector_')
val_speaker_labels_prefix = os.path.join(dataset_dir, 'val/label_')
val_angle_labels_prefix = os.path.join(dataset_dir, 'val/label2_')
val_idx_file = os.path.join(dataset_dir, 'val/idx.npy')

# Output directories
models_dir = os.path.join(workspace_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

plots_dir = os.path.join(workspace_dir, 'plots')
plot_folder = os.path.join(plots_dir, 'confusion_matrices')
os.makedirs(plot_folder, exist_ok=True)

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
plt.close("all")                                                                                                           
batch_size = 64
num_speaker_classes = 3
num_angle_classes = 20
epochs = 20

# Audio feature dimensions
num_freq_bins = 1025
num_mic_channels = 7

current_day = strftime("%d", gmtime())
current_month = strftime("%m", gmtime())
kernel_norm_limit = max_norm(5.0)

# ---------------------------------------------------------
# Data Generator
# ---------------------------------------------------------
class DataGenerator(Sequence):
    def __init__(self, data_prefix, speaker_label_prefix, angle_label_prefix, idx_file, 
                 batch_size=batch_size, dim=(num_freq_bins, num_mic_channels),
                 num_speaker_classes=num_speaker_classes, num_angle_classes=num_angle_classes):
        
        self.data_prefix = data_prefix
        self.speaker_label_prefix = speaker_label_prefix
        self.angle_label_prefix = angle_label_prefix
        self.batch_size = batch_size
        self.dim = dim
        self.num_speaker_classes = num_speaker_classes
        self.num_angle_classes = num_angle_classes
        
        self.total_samples = np.load(idx_file)
        self.on_epoch_end()

    def __len__(self):
        return int(self.total_samples / self.batch_size)
    
    def __getitem__(self, index):
        # Calculate start and end indices for the current batch
        batch_indices = self.dataset_indices[index * self.batch_size : (index + 1) * self.batch_size]
        X, y_speaker, y_angle = self.__data_generation(batch_indices)
        return X, [y_speaker, y_angle]

    def on_epoch_end(self):
        self.dataset_indices = np.arange(self.total_samples)
        np.random.shuffle(self.dataset_indices)

    def __data_generation(self, batch_indices):
        X_batch = np.empty((self.batch_size, *self.dim))
        speaker_labels_batch = np.empty((self.batch_size), dtype=int)
        angle_labels_batch = np.empty((self.batch_size), dtype=int)
        
        for i, sample_idx in enumerate(batch_indices):
            X_batch[i,] = np.load(self.data_prefix + str(sample_idx) + '.npy')
            speaker_labels_batch[i] = np.load(self.speaker_label_prefix + str(sample_idx) + '.npy')
            angle_labels_batch[i] = np.load(self.angle_label_prefix + str(sample_idx) + '.npy')
            
        return X_batch, to_categorical(speaker_labels_batch, num_classes=self.num_speaker_classes), \
                        to_categorical(angle_labels_batch, num_classes=self.num_angle_classes)
  
# ---------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------
inputs = Input(shape=[num_freq_bins, num_mic_channels])

# Shared Convolutional Blocks
x = Conv1D(filters=64, kernel_size=6, strides=2, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(inputs)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(filters=64, kernel_size=3, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(filters=32, kernel_size=3, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(filters=32, kernel_size=3, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(128, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
shared_features = Dropout(0.25)(x)

# Head 1: Angle / Direction of Arrival (20 classes)
angle_fc = Dense(32, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(shared_features)
angle_fc = Activation('relu')(angle_fc)
angle_fc = Dropout(0.25)(angle_fc)
out_angle = Dense(20, activation='softmax', name='out_angle')(angle_fc)

# Head 2: Speaker Count (3 classes)
speaker_fc = Dense(10, kernel_constraint=kernel_norm_limit, bias_constraint=kernel_norm_limit)(shared_features)
speaker_fc = Activation('relu')(speaker_fc)
out_speaker_count = Dense(3, activation='softmax', name='out_speaker_count')(speaker_fc)

# Compile Main Multi-Output Model
full_model = Model(inputs=inputs, outputs=[out_speaker_count, out_angle])
full_model.summary()

loss_weights = {"out_speaker_count": 1.0, "out_angle": 2.0}   
full_model.compile(
    loss=[OCC3.loss33, OCC.loss1], 
    loss_weights=loss_weights,
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Sub-models for inference saving
model_speaker_count = Model(inputs=inputs, outputs=out_speaker_count)
model_angle = Model(inputs=inputs, outputs=out_angle)

# ---------------------------------------------------------
# Training Initialization
# ---------------------------------------------------------
training_generator = DataGenerator(
    train_data_prefix, 
    train_speaker_labels_prefix,
    train_angle_labels_prefix,
    train_idx_file)

validation_generator = DataGenerator(
    val_data_prefix, 
    val_speaker_labels_prefix,
    val_angle_labels_prefix,
    val_idx_file)

print(f"Total training samples: {training_generator.total_samples}")
print(f"Total training batches per epoch: {len(training_generator)}")

X_batch, y_batch = training_generator[0]
print(f"Input X shape: {X_batch.shape}")

y_speaker_batch, y_angle_batch = y_batch
print(f"Label speaker shape (3 classes): {y_speaker_batch.shape}")
print(f"Label angle shape (20 classes): {y_angle_batch.shape}")

# ---------------------------------------------------------
# Execute Training
# ---------------------------------------------------------
history = full_model.fit(
    x=training_generator,
    validation_data=validation_generator,
    epochs=epochs,
    workers=12,
    verbose=2,
)

# Save Models
speaker_model_path = os.path.join(models_dir, f'model_speaker_GEVD_{current_day}_{current_month}.h5')
model_speaker_count.save(speaker_model_path)

angle_model_path = os.path.join(models_dir, f'model_angle_GEVD_{current_day}_{current_month}.h5')
model_angle.save(angle_model_path)

# ---------------------------------------------------------
# Evaluation & Confusion Matrices
# ---------------------------------------------------------
val_size = np.load(val_idx_file)

val_features = np.zeros((val_size, num_freq_bins, num_mic_channels), dtype=float)
true_speaker_counts = np.zeros(val_size)
true_angles = np.zeros(val_size)

for index in range(val_size):
    val_features[index,:,:] = np.squeeze(np.load(val_data_prefix + str(index) + '.npy'))
    true_speaker_counts[index] = np.load(val_speaker_labels_prefix + str(index) + '.npy')
    true_angles[index] = np.load(val_angle_labels_prefix + str(index) + '.npy')
        
# Plot settings
annot = True
cmap = 'Oranges'
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
fz = 9
figsize = [18, 18]

# --- Speaker Count Evaluation ---
pred_speaker_probs = model_speaker_count.predict(val_features)
pred_speaker_df = pd.DataFrame(pred_speaker_probs)
pred_speaker_classes = pred_speaker_df.idxmax(axis=1)

speaker_num_classes = model_speaker_count.layers[-1].output_shape[1]     
speaker_plot_labels = ['Noise', 'One speaker', '2 speakers']

plot_confusion_matrix_from_data(
    true_speaker_counts, pred_speaker_classes, speaker_num_classes, speaker_plot_labels,
    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
    name='confusion_matrix_3_classes.png', plot_folder=plot_folder
)

# --- Angle / DOA Evaluation ---
pred_angle_probs = model_angle.predict(val_features)
pred_angle_df = pd.DataFrame(pred_angle_probs)
pred_angle_classes = pred_angle_df.idxmax(axis=1)

# Filter out classes 0 and 19 for the confusion matrix display
filtered_true_angles = np.copy(true_angles)
filtered_pred_angles = pred_angle_classes.to_numpy()

filtered_pred_angles = np.delete(filtered_pred_angles, np.where(filtered_true_angles == 0)[0])
filtered_true_angles = np.delete(filtered_true_angles, np.where(filtered_true_angles == 0))

filtered_pred_angles = np.delete(filtered_pred_angles, np.where(filtered_true_angles == 19))
filtered_true_angles = np.delete(filtered_true_angles, np.where(filtered_true_angles == 19))

angle_num_classes = model_angle.layers[-1].output_shape[1] - 2    
angle_plot_labels = [
    '0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100',
    '101-110', '111-120', '121-130', '131-140', '141-150', '151-160', '161-170', '171-180'
]

plot_confusion_matrix_from_data(
    filtered_true_angles - 1, filtered_pred_angles - 1, angle_num_classes, angle_plot_labels,
    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
    name='confusion_matrix_18_classes.png', plot_folder=plot_folder
)


# ---------------------------------------------------------
# Plot Training History (Loss Graph)
# ---------------------------------------------------------
# Create a new figure
plt.figure(figsize=(14, 8))

# 1. Plot the Total Combined Loss (Solid lines)
plt.plot(history.history['loss'], label='Total Training Loss', color='black', linewidth=2.5)
plt.plot(history.history['val_loss'], label='Total Validation Loss', color='red', linewidth=2.5)

# 2. Plot the Sub-Losses for Speaker Count (Dashed lines)
plt.plot(history.history['out_speaker_count_loss'], label='Train Speaker Loss', linestyle='--', color='blue', alpha=0.7)
plt.plot(history.history['val_out_speaker_count_loss'], label='Val Speaker Loss', linestyle='--', color='lightblue', alpha=0.7)

# 3. Plot the Sub-Losses for Angle (Dotted lines)
plt.plot(history.history['out_angle_loss'], label='Train Angle Loss', linestyle=':', color='green', alpha=0.7)
plt.plot(history.history['val_out_angle_loss'], label='Val Angle Loss', linestyle=':', color='lightgreen', alpha=0.7)

# Format the graph
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss (Categorical Crossentropy)')
plt.xlabel('Epoch')
plt.xticks(np.arange(epochs)) # Force x-axis to show whole integer epochs

# Put a legend to the right of the current axis
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot to your plots directory
loss_plot_path = os.path.join(plots_dir, 'training_loss_graph.png')
plt.savefig(loss_plot_path, bbox_inches='tight')
print(f"Loss graph saved to: {loss_plot_path}")