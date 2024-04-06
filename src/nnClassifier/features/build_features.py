import numpy as np

from nnClassifier import logger

def normalize_splits(splits):
    mean = np.reshape(np.mean(splits["train"][0], axis = 1), (-1,1))
    std = np.reshape(np.std(splits["train"][0], axis = 1), (-1, 1))
    def _normalize(X, mean, std):
        return (X - mean)/std
    splits_norm = {k: (_normalize(v[0], mean, std), v[1], v[2]) for (k,v) in splits.items()}
    logger.info("Splits successfully normalized.")
    return splits_norm