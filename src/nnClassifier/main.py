import numpy as np

from nnClassifier import logger

from nnClassifier.data.make_dataset import make_splits
from nnClassifier.features.build_features import normalize_splits
from nnClassifier.models.train_model import OneLayerClassifier, TwoLayerClassifier
from nnClassifier.models.predict_model import predict

def main(seed):
    splits_norm = normalize_splits(make_splits())
    X_train, Y_train, y_train = splits_norm["train"]
    X_val, Y_val, y_val = splits_norm["validation"]
    X_test, Y_test, y_test = splits_norm["test"]

    np.random.seed(seed)


if __name__ == '__main__':
    main()