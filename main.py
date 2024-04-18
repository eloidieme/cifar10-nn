from pathlib import Path
import argparse
from time import time

import numpy as np

from nnClassifier import logger

from nnClassifier.data.make_dataset import make_splits_full
from nnClassifier.features.build_features import normalize_splits, augment_data
from nnClassifier.models.train import Model, N_HIDDEN_NODES
from nnClassifier.models.train_model import OneLayerClassifier
from nnClassifier.models.predict import predict

def main(seed):
    np.random.seed(seed)

    splits_norm = normalize_splits(make_splits_full("test_batch", 1000))
    X_train, Y_train, y_train = splits_norm["train"]
    X_val, Y_val, y_val = splits_norm["validation"]
    X_test, Y_test, y_test = splits_norm["test"]
    X_aug, Y_aug, y_aug = augment_data(X_train, Y_train, y_train) 
    print(X_aug.shape[1])

    parser = argparse.ArgumentParser(description='Argument parser for training.')

    parser.add_argument('-nl', '--n-layers', dest='n_layers', type=int, default=2, help='Specify number of layers.')
    parser.add_argument('-nb', '--n-batches', dest='n_batch', type=int, default=200, help='Specify number of samples in one batch.')
    parser.add_argument('-ne', '--n-epochs', dest='n_epochs', type=int, default=80, help='Specify number of epochs.')
    parser.add_argument('-e', '--eta', dest='eta', type=float, default=0.00001, help='Specify value of learning rate eta.')
    parser.add_argument('-l', '--lambda', dest='lamda', type=float, default=0.01, help='Specify value of regularization factor lambda.')
    parser.add_argument('-clr', '--cyclical', dest='cyclical_lr', action='store_true', help='Run training with cyclical learning rate.')
    parser.add_argument('-dr', '--dropout', dest='dropout', type=float, default=None, help='Run training with dropout.')
    
    args = parser.parse_args()

    gd_params = {"n_batch": args.n_batch, "n_epochs": args.n_epochs, "eta": args.eta, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8}
    lamda = args.lamda

    logger.info("Data loaded for training.")

    figure_savepath = Path(f"./reports/figures/tc_{args.n_layers}-{N_HIDDEN_NODES}-{gd_params['n_batch']}-{gd_params['n_epochs']}-{gd_params['eta']}-{lamda}-{args.dropout if args.dropout else 0}.png")

    if args.n_layers == 1:
        model_savepath = f'./models/oneLayerNN_{time()}'
        model = OneLayerClassifier(X_aug, Y_aug, gd_params, lamda=lamda, cyclical_lr = args.cyclical_lr, validation=(X_val, Y_val, y_val), seed=seed)
    elif args.n_layers == 2:
        model_savepath = f'./models/twoLayersNN_{time()}'
        model = Model(X_aug, Y_aug, gd_params, dropout_rate=args.dropout, lamda=lamda, cyclical_lr = args.cyclical_lr, validation=(X_val, Y_val, y_val), seed=seed)
    logger.info("Model created.")

    logger.info("Start of main process.")
    model.run_training(gd_params, figure_savepath=figure_savepath, test_data=(X_test, y_test), model_savepath=model_savepath)
    logger.info("End of main process.")

if __name__ == '__main__':
    main(400)