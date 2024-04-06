import numpy as np

def normalize_splits(splits):
    mean = np.reshape(np.mean(splits["train"][0], axis = 1), (-1,1))
    std = np.reshape(np.std(splits["train"][0], axis = 1), (-1, 1))
    def _normalize(X, mean, std):
        return (X - mean)/std
    splits_norm = {k: (_normalize(v[0], mean, std), v[1], v[2]) for (k,v) in splits.items()}
    return splits_norm