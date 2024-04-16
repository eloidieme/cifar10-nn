# -*- coding: utf-8 -*-
import os
from pathlib import Path
import subprocess
import pickle

from dotenv import find_dotenv, load_dotenv
import numpy as np

from nnClassifier import logger

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = f"{PROJECT_DIR}/data"
DATASET_ARCHIVE = "cifar-10-python.tar.gz"
DATASET_DIR = "cifar-10-batches-py"
DATASET_URL = f"https://www.cs.toronto.edu/~kriz/{DATASET_ARCHIVE}"

def get_raw_data():
    """
    Gets the data from the CIFAR-10 website.
    """
    try:
        if not DATASET_ARCHIVE in os.listdir(DATA_DIR):
            subprocess.run([
                "wget", 
                "-P", 
                DATA_DIR, 
                DATASET_URL,
            ])
            subprocess.run([
                "tar", 
                "-xzf",
                f"{DATA_DIR}/{DATASET_ARCHIVE}",
                "-C",
                DATA_DIR
            ])
            logger.info("Raw data successfully downloaded and extracted.")
        elif not DATASET_DIR in os.listdir(DATA_DIR):
            subprocess.run([
                "tar", 
                "-xzf",
                f"{DATA_DIR}/{DATASET_ARCHIVE}",
                "-C",
                DATA_DIR
            ])
            logger.info("Raw data successfully extracted.")
        else:
            logger.info("Raw data already available. Remove it to download and extract again.")
    except Exception as e:
        logger.error("Error occured while downloading raw data: %s", e)

def load_data(filename):
    with open('data/cifar-10-batches-py/'+filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X, y = np.array(data[b'data']).T, np.array(data[b'labels'], dtype=np.int8).T
    Y = []
    for label in data[b'labels']:
        one_hot = np.zeros(10)
        one_hot[label] = 1
        Y.append(one_hot)
    Y = np.array(Y, dtype=np.int8).T
    logger.info("Data from %s correctly loaded.", filename)
    return X, Y, y

def make_splits(train_data, val_data, test_data):
    splits = {}
    splits["train"] = load_data(train_data)
    splits["validation"] = load_data(val_data)
    splits["test"] = load_data(test_data)
    logger.info("Split of train, validation and test data successfull.")
    return splits

def make_splits_full(test_data, val_size = 1000):
    splits = {}
    for i in range(5):
        data = load_data(f"data_batch_{i+1}")
        if i == 0:
            X, Y, y = data
        else:
            X, Y, y = np.concatenate((X, data[0]), axis=1), np.concatenate((Y, data[1]), axis=1), np.concatenate((y, data[2])), 
    n = X.shape[1]
    perm = np.random.permutation(n)
    X_train, X_val = X[:, perm][:, val_size:], X[:, perm][:, :val_size]
    Y_train, Y_val = Y[:, perm][:, val_size:], Y[:, perm][:, :val_size]
    y_train, y_val = y[perm][val_size:], y[perm][:val_size]
    splits["train"] = (X_train, Y_train, y_train)
    splits["validation"] = (X_val, Y_val, y_val)
    splits["test"] = load_data(test_data)
    logger.info("Split of train, validation and test data using full data successfull.")
    return splits

if __name__ == '__main__':
    get_raw_data()