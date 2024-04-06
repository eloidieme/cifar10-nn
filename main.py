import sys
from pathlib import Path

import numpy as np

from nnClassifier import logger

from nnClassifier.data.make_dataset import make_splits
from nnClassifier.features.build_features import normalize_splits
from nnClassifier.models.train_model import OneLayerClassifier, TwoLayerClassifier
from nnClassifier.models.predict_model import predict

def main(seed):
    np.random.seed(seed)

    splits_norm = normalize_splits(make_splits("data_batch_1", "data_batch_2", "test_batch"))
    X_train, Y_train, _ = splits_norm["train"]
    X_val, Y_val, y_val = splits_norm["validation"]
    X_test, _, y_test = splits_norm["test"]

    gd_params = {"n_batch": int(sys.argv[1]), "n_epochs": int(sys.argv[2]), "eta": float(sys.argv[3])}
    lamda = float(sys.argv[4])

    savepath = f"./reports/figures/training_curves_{gd_params['n_batch']}_{gd_params['n_epochs']}_{gd_params['eta']}_{lamda}.png"

    model = TwoLayerClassifier(X_train, Y_train, gd_params, lamda=lamda, validation=(X_val, Y_val, y_val), seed=seed)
    model.run_training(gd_params, savepath, test_data=(X_test, y_test))
    #print(f"Maximum difference between numerical and analytical gradients: {model.validate_gradient(X_train[:50, :20], Y_train[:50, :20], h=1e-5, eps=1e-10)}")

if __name__ == '__main__':
    main(400)