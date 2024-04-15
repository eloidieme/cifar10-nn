import sys
from pathlib import Path
import argparse
from time import time

import numpy as np

from nnClassifier import logger

from nnClassifier.data.make_dataset import make_splits, make_splits_full
from nnClassifier.features.build_features import normalize_splits, augment_data
from nnClassifier.models.train_model import OneLayerClassifier, TwoLayerClassifier
from nnClassifier.models.predict import predict

def main(seed):
    np.random.seed(seed)

    splits_norm = normalize_splits(make_splits("data_batch_1", "data_batch_2", "test_batch"))
    splits_norm = normalize_splits(make_splits_full("test_batch", 1000))
    X_train, Y_train, y_train = splits_norm["train"]
    X_val, Y_val, y_val = splits_norm["validation"]
    X_test, Y_test, y_test = splits_norm["test"]

    X_aug, Y_aug, y_aug = augment_data(X_train, Y_train, y_train)

    """ fd = np.load('./data/full_data_augmented.npz', 'r+')
    X_aug, Y_aug, y_aug = fd["X_train"], fd["Y_train"], fd["y_train"]
    X_val, Y_val, y_val = fd["X_val"], fd["Y_val"], fd["y_val"]
    X_test, Y_test, y_test = fd["X_test"], fd["Y_test"], fd["y_test"] """

    parser = argparse.ArgumentParser(description='Argument parser for training.')

    parser.add_argument('-nl', '--n-layers', dest='n_layers', type=int, default=1, help='Specify number of layers.')
    parser.add_argument('-nb', '--n-batches', dest='n_batch', type=int, default=100, help='Specify number of samples in one batch.')
    parser.add_argument('-ne', '--n-epochs', dest='n_epochs', type=int, default=40, help='Specify number of epochs.')
    parser.add_argument('-e', '--eta', dest='eta', type=float, default=0.001, help='Specify value of learning rate eta.')
    parser.add_argument('-l', '--lambda', dest='lamda', type=float, default=0.0, help='Specify value of regularization factor lambda.')
    parser.add_argument('-clr', '--cyclical', dest='cyclical_lr', action='store_true', help='Run training with cyclical learning rate.')
    
    args = parser.parse_args()

    gd_params = {"n_batch": args.n_batch, "n_epochs": args.n_epochs, "eta": args.eta}
    lamda = args.lamda

    logger.info("Data loaded for training.")

    figure_savepath = Path(f"./reports/figures/tc_{args.n_layers}-{gd_params['n_batch']}-{gd_params['n_epochs']}-{gd_params['eta']}-{lamda}.png")

    if args.n_layers == 1:
        model_savepath = f'./models/oneLayerNN_{time()}'
        model = OneLayerClassifier(X_aug, Y_aug, gd_params, lamda=lamda, cyclical_lr = args.cyclical_lr, validation=(X_val, Y_val, y_val), seed=seed)
    elif args.n_layers == 2:
        model_savepath = f'./models/twoLayersNN_{time()}'
        model = TwoLayerClassifier(X_aug, Y_aug, gd_params, lamda=lamda, cyclical_lr = args.cyclical_lr, validation=(X_val, Y_val, y_val), seed=seed)
    logger.info("Model created.")

    logger.info("Start of main process.")
    model.run_training(gd_params, figure_savepath=figure_savepath, test_data=(X_test, y_test), model_savepath=model_savepath)
    logger.info("End of main process.")

if __name__ == '__main__':
    main(400)