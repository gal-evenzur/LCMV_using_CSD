# -*- coding: utf-8 -*-
"""
Refactored Multi-Output Model Training Script
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime

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
from keras.constraints import max_norm

# Custom Dependencies
from plot_confusion_matrix_from_data import plot_confusion_matrix_from_data
import ordinal_categorical_crossentropy18 as OCC
import ordinal_categorical_crossentropy3 as OCC3


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
def load_config():
    """Centralizes all hyperparameters, dimensions, and paths."""
    pydir = os.path.dirname(os.path.realpath(__file__))
    workspace_dir = os.path.dirname(pydir)
    dataset_dir = os.path.join(workspace_dir, 'data', 'dataset')
    
    current_day = strftime("%d", gmtime())
    current_month = strftime("%m", gmtime())
    
    config = {
        "batch_size": 64,
        "epochs": 20,
        "num_freq_bins": 1025,
        "num_mic_channels": 7,
        "num_speaker_classes": 3,
        "num_angle_classes": 20,
        "learning_rate": 0.001,
        "kernel_norm_limit": max_norm(5.0),
        "timestamp": f"{current_day}_{current_month}",
        "paths": {
            "train_prefix": os.path.join(dataset_dir, 'train/feature_vector_'),
            "train_speaker_prefix": os.path.join(dataset_dir, 'train/label_'),
            "train_angle_prefix": os.path.join(dataset_dir, 'train/label2_'),
            "train_idx": os.path.join(dataset_dir, 'train/idx.npy'),
            
            "val_prefix": os.path.join(dataset_dir, 'val/feature_vector_'),
            "val_speaker_prefix": os.path.join(dataset_dir, 'val/label_'),
            "val_angle_prefix": os.path.join(dataset_dir, 'val/label2_'),
            "val_idx": os.path.join(dataset_dir, 'val/idx.npy'),
            
            "models_dir": os.path.join(workspace_dir, 'models'),
            "plots_dir": os.path.join(workspace_dir, 'plots'),
            "confusion_matrices_dir": os.path.join(workspace_dir, 'plots', 'confusion_matrices')
        }
    }
    
    # Ensure output directories exist
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    os.makedirs(config["paths"]["confusion_matrices_dir"], exist_ok=True)
    
    return config


# ---------------------------------------------------------
# 2. Data Generator
# ---------------------------------------------------------
class DataGenerator(Sequence):
    def __init__(self, data_prefix, speaker_label_prefix, angle_label_prefix, idx_file, config):
        self.data_prefix = data_prefix
        self.speaker_label_prefix = speaker_label_prefix
        self.angle_label_prefix = angle_label_prefix
        self.batch_size = config["batch_size"]
        self.dim = (config["num_freq_bins"], config["num_mic_channels"])
        self.num_speaker_classes = config["num_speaker_classes"]
        self.num_angle_classes = config["num_angle_classes"]
        
        self.total_samples = np.load(idx_file)
        self.on_epoch_end()

    def __len__(self):
        return int(self.total_samples / self.batch_size)
    
    def __getitem__(self, index):
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


def get_data_generators(config):
    """Instantiates and returns the training and validation generators."""
    paths = config["paths"]
    
    train_gen = DataGenerator(
        paths["train_prefix"], paths["train_speaker_prefix"], 
        paths["train_angle_prefix"], paths["train_idx"], config
    )
    
    val_gen = DataGenerator(
        paths["val_prefix"], paths["val_speaker_prefix"], 
        paths["val_angle_prefix"], paths["val_idx"], config
    )
    
    return train_gen, val_gen


# ---------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------
def build_multi_output_model(config):
    """Builds and compiles the shared CNN and multi-head outputs."""
    inputs = Input(shape=[config["num_freq_bins"], config["num_mic_channels"]])
    knl = config["kernel_norm_limit"]

    # Shared Convolutional Blocks
    x = Conv1D(filters=64, kernel_size=6, strides=2, kernel_constraint=knl, bias_constraint=knl)(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=3, kernel_constraint=knl, bias_constraint=knl)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=32, kernel_size=3, kernel_constraint=knl, bias_constraint=knl)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=32, kernel_size=3, kernel_constraint=knl, bias_constraint=knl)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, kernel_constraint=knl, bias_constraint=knl)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    shared_features = Dropout(0.25)(x)

    # Head 1: Angle / Direction of Arrival
    angle_fc = Dense(32, kernel_constraint=knl, bias_constraint=knl)(shared_features)
    angle_fc = Activation('relu')(angle_fc)
    angle_fc = Dropout(0.25)(angle_fc)
    out_angle = Dense(config["num_angle_classes"], activation='softmax', name='out_angle')(angle_fc)

    # Head 2: Speaker Count
    speaker_fc = Dense(10, kernel_constraint=knl, bias_constraint=knl)(shared_features)
    speaker_fc = Activation('relu')(speaker_fc)
    out_speaker_count = Dense(config["num_speaker_classes"], activation='softmax', name='out_speaker_count')(speaker_fc)

    # Compile Main Multi-Output Model
    full_model = Model(inputs=inputs, outputs=[out_speaker_count, out_angle])
    
    loss_weights = {"out_speaker_count": 1.0, "out_angle": 2.0}   
    full_model.compile(
        loss=[OCC3.loss33, OCC.loss1], 
        loss_weights=loss_weights,
        optimizer=Adam(learning_rate=config["learning_rate"]),
        metrics=['accuracy']
    )

    # Sub-models for inference saving
    model_speaker_count = Model(inputs=inputs, outputs=out_speaker_count)
    model_angle = Model(inputs=inputs, outputs=out_angle)
    
    return full_model, model_speaker_count, model_angle


# ---------------------------------------------------------
# 4. Training & Plotting Logic
# ---------------------------------------------------------
def train_and_save(config, full_model, model_speaker, model_angle, train_gen, val_gen):
    """Executes the training loop and saves the independent sub-models."""
    print(f"Total training samples: {train_gen.total_samples}")
    print(f"Total training batches per epoch: {len(train_gen)}")

    # Enforcing serial execution
    history = full_model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=config["epochs"],
        workers=1, 
        use_multiprocessing=False, 
        verbose=2,
    )

    # Save Models
    ts = config["timestamp"]
    models_dir = config["paths"]["models_dir"]
    
    model_speaker.save(os.path.join(models_dir, f'model_speaker_GEVD_{ts}.h5'))
    model_angle.save(os.path.join(models_dir, f'model_angle_GEVD_{ts}.h5'))
    
    return history

def plot_training_history(config, history):
    """Plots and saves the training loss curves."""
    plt.figure(figsize=(14, 8))
    
    plt.plot(history.history['loss'], label='Total Training Loss', color='black', linewidth=2.5)
    plt.plot(history.history['val_loss'], label='Total Validation Loss', color='red', linewidth=2.5)
    
    plt.plot(history.history['out_speaker_count_loss'], label='Train Speaker Loss', linestyle='--', color='blue', alpha=0.7)
    plt.plot(history.history['val_out_speaker_count_loss'], label='Val Speaker Loss', linestyle='--', color='lightblue', alpha=0.7)
    
    plt.plot(history.history['out_angle_loss'], label='Train Angle Loss', linestyle=':', color='green', alpha=0.7)
    plt.plot(history.history['val_out_angle_loss'], label='Val Angle Loss', linestyle=':', color='lightgreen', alpha=0.7)

    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(config["epochs"]))
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    loss_plot_path = os.path.join(config["paths"]["plots_dir"], 'training_loss_graph.png')
    plt.savefig(loss_plot_path, bbox_inches='tight')
    print(f"Loss graph saved to: {loss_plot_path}")


def evaluate_models(config, model_speaker, model_angle):
    """Loads validation data, runs inference, and generates confusion matrices."""
    paths = config["paths"]
    val_size = np.load(paths["val_idx"])

    val_features = np.zeros((val_size, config["num_freq_bins"], config["num_mic_channels"]), dtype=float)
    true_speaker_counts = np.zeros(val_size)
    true_angles = np.zeros(val_size)

    for index in range(val_size):
        val_features[index,:,:] = np.squeeze(np.load(paths["val_prefix"] + str(index) + '.npy'))
        true_speaker_counts[index] = np.load(paths["val_speaker_prefix"] + str(index) + '.npy')
        true_angles[index] = np.load(paths["val_angle_prefix"] + str(index) + '.npy')
        
    # Plot settings
    annot, cmap, fmt, lw, cbar = True, 'Oranges', '.2f', 0.5, False
    show_null_values, pred_val_axis, fz, figsize = 2, 'y', 9, [18, 18]

    # --- Speaker Count Evaluation ---
    pred_speaker_probs = model_speaker.predict(val_features)
    pred_speaker_classes = np.argmax(pred_speaker_probs, axis=1)

    speaker_num_classes = model_speaker.layers[-1].output_shape[1]     
    speaker_plot_labels = ['Noise', 'One speaker', '2 speakers']

    plot_confusion_matrix_from_data(
        true_speaker_counts, pred_speaker_classes, speaker_num_classes, speaker_plot_labels,
        annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis,
        name='confusion_matrix_3_classes.png', plot_folder=paths["confusion_matrices_dir"]
    )

    # --- Angle / DOA Evaluation ---
    pred_angle_probs = model_angle.predict(val_features)
    pred_angle_classes = np.argmax(pred_angle_probs, axis=1)

    filtered_true_angles = np.copy(true_angles)
    filtered_pred_angles = pred_angle_classes

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
        name='confusion_matrix_18_classes.png', plot_folder=paths["confusion_matrices_dir"]
    )


# ---------------------------------------------------------
# 5. Main Execution Block
# ---------------------------------------------------------
def main():
    plt.close("all") 
    
    config = load_config()
    train_gen, val_gen = get_data_generators(config)
    
    full_model, model_speaker, model_angle = build_multi_output_model(config)
    full_model.summary()
    
    history = train_and_save(config, full_model, model_speaker, model_angle, train_gen, val_gen)
    
    plot_training_history(config, history)
    evaluate_models(config, model_speaker, model_angle)

if __name__ == "__main__":
    main()
