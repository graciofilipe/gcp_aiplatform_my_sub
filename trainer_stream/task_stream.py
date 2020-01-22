import datetime
import json
import logging
import os

import numpy as np
import pandas as pd
from pandas.compat import StringIO
from tensorflow.python.lib.io import file_io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tensorflow import feature_column
from tensorflow.keras import layers

"""
This script fits two models to processed data, writes performance info for both models, and saves the larger one
"""

np.random.seed(111)

TRAINING_EPOCHS = 11
LABEL = 'purchased'
INPUT_FOLDER = 'processed_data'
OUTPUT_FOLDER = 'modeling_output'
DATA_DIR = 'gs://aiplatformfilipegracio2020/data'


def df_to_pred_dataset(dataframe,  batch_size=1024):
    """
    A utility function to create a tf.data dataset from a Pandas Dataframe
    :param dataframe: pandas data frame to convert
    :param target_col: str, the label to predict
    :param shuffle: bool
    :param batch_size: int
    :return:
    """
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    ds = ds.batch(batch_size)
    return ds


def df_to_train_dataset(dataframe, target_col, shuffle=True, batch_size=1024):
    """
    A utility function to create a tf.data dataset from a Pandas Dataframe
    :param dataframe: pandas data frame to convert
    :param target_col: str, the label to predict
    :param shuffle: bool
    :param batch_size: int
    :return:
    """
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def calculate_class_weights(label_data):
    """
    class weights inverse to their frequency in the data
    :param label_data: a data frame column containing the label data
    :return: dictionary of class weights
    """
    neg, pos = np.bincount(label_data)
    weight_for_0 = 1 / neg
    weight_for_1 = 1 / pos
    return {0: weight_for_0, 1: weight_for_1}


def make_tf_datasets(data_df, target_col):
    """
    prepares the tensorflow train, validation, and test datasets from a pandas dataframe
    :param data_df: pandas datafarme
    :param target_col: str, the column to predict
    :return: (tf.Dataset, tf.Dataset, tf.Dataset) train, validation and test tensorflow datasets
    """
    train, test = train_test_split(data_df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    print(len(train), 'train examples')
    print(len(test), 'test examples')

    batch_size = 1024
    train_ds = df_to_train_dataset(
        train, target_col, shuffle=True, batch_size=batch_size)
    val_ds = df_to_train_dataset(
        val, target_col, shuffle=False, batch_size=batch_size)
    test_ds = df_to_train_dataset(
        test, target_col, shuffle=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds


def stream_to_dataset(features_stream, labels_stream):
    logging.info('INTO THE STREAMING FUNCTION')
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(features_stream), labels_stream))
    return ds


def make_simple_feature_layer(data):
    """
    given the dataset, prepares a simple input layer to the tensorflow model
    :param data: pandas dataset to be used in the modeling
    :return: a tensorflow Features Layer object to serve as model layer
    """

    feature_columns = []

    # numeric cols
    for numeric_col in ['price']:
        feature_columns.append(feature_column.numeric_column(numeric_col))

    return tf.keras.layers.DenseFeatures(feature_columns)


def make_simple_model(feature_layer):
    """
    creates the tensorflow simple model equivalent to a logistic regression
    :param feature_layer: keras feature layer
    :return: tensroflow model ready for fitting
    """
    model = tf.keras.Sequential([feature_layer,
                                 layers.Dense(1, activation='sigmoid')
                                 ])

    metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=metrics)
    return model


def model_fit_and_evaluate(model, train_ds, job_name, model_saving=False):
    """
    fits the model, saves it, and returns model performance info
    :param model: tensorflow model to train
    :param train_ds: tf.Dataset of training samples
    :param val_ds: tf.Dataset of validation samples
    :param test_ds: tf.Dataset of test samples
    :param epochs: the number of epochs to run through the data
    :param class_weights: dict containing the class weights
    :param job_name: str of the job model fitting job name
    :param model_saving: boolean, allows for the model to be saved in h5 format
    :return: dictionary of model performance and run info
    """

    # to avoid overfitting stop when validation loss doesn't improve further
    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=6)
    model.fit(train_ds, verbose=1, epochs=epochs)

    return model


def get_data():
    # Access iris data from Cloud Storage
    logging.info('GETTING THE FEATURES DATA')
    features_filesteam = file_io.FileIO(os.path.join(DATA_DIR, 'price.csv'),
                                        mode='r')
    features = pd.read_csv(
        StringIO(features_filesteam.read())).values

    logging.info('GETTING THE LABELS DATA')
    target_filestream = file_io.FileIO(os.path.join(DATA_DIR, 'target.csv'),
                                       mode='r')
    target = pd.read_csv(StringIO(target_filestream.read())).values
    target = target.reshape((target.size,))

    logging.info('FINISHING THE DATA')

    return target, features


def main_modeling_pipeline():
    """
    runs the full modeling pipeline
    """
    logging.info('STARTING THE MAIN')

    # getdata
    target, features = get_data()
    logging.info('Data loaded and processed')

    ds = stream_to_dataset(features_stream=features, labels_stream=target)
    logging.info('Tensorflow datasets created')

    simple_feature_layer = make_simple_feature_layer(data_df)
    simple_model = make_simple_model(simple_feature_layer)
    simple_model_results, simple_model = model_fit_and_evaluate(model=simple_model,
                                                                train_ds=train_ds,
                                                                epochs=TRAINING_EPOCHS,
                                                                job_name='simple_model')

    simple_model.save('gs://aiplatformfilipegracio2020/')
    iris_data_filesteam.close()
    target_filestream.close()


if __name__ == '__main__':
    main_modeling_pipeline()
