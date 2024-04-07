from pathlib import Path
import argparse

import numpy as np

from nnClassifier.data.make_dataset import make_splits_full
from nnClassifier.features.build_features import normalize_splits
from nnClassifier.models.train_model import TwoLayerClassifier

def main(seed):
    np.random.seed(seed)

    splits_norm = normalize_splits(make_splits_full('test_batch', 5000))
    X_train, Y_train, _ = splits_norm["train"]
    X_val, Y_val, y_val = splits_norm["validation"]
    X_test, _, y_test = splits_norm["test"]

    parser = argparse.ArgumentParser(description='Argument parser for training.')

    parser.add_argument('-nb', '--n-batches', dest='n_batch', type=int, default=100, help='Specify number of samples in one batch.')
    parser.add_argument('-ne', '--n-epochs', dest='n_epochs', type=int, default=40, help='Specify number of epochs.')
    
    args = parser.parse_args()

    gd_params = {"n_batch": args.n_batch, "n_epochs": args.n_epochs, "eta": 0}
    l_min, l_max = -5, -1
    lambdas = []
    for _ in range(8):
        l = l_min + (l_max - l_min)*np.random.uniform()
        lambdas.append(10**l)

    for lamda in lambdas:
        savepath = Path(f"./reports/figures/lgd/tc_2-{gd_params['n_batch']}-{gd_params['n_epochs']}-cyclical-{lamda}.png")
        model = TwoLayerClassifier(X_train, Y_train, gd_params, lamda=lamda, cyclical_lr = True, validation=(X_val, Y_val, y_val), seed=seed)
        model.run_training(gd_params, savepath, test_data=(X_test, y_test))

if __name__ == '__main__':
    main(400)