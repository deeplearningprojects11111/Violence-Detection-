import numpy as np
from keras.preprocessing.image import ImageDataGenerator #used for image augmentation and preprocessing during deep learning model training
from keras.layers import Dense, Flatten, conv2D, MaxPooling2O ,Drouput
from tensorflow.keras.models import Model
import os
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
import mlflow.keras

def train_model_mlflow(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['image_size']
        train_set = config['model']['train_path']
        test_set = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertifal_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']
        model_path = config['model']['sav_dir']