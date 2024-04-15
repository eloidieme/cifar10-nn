import numpy as np
import cv2

from nnClassifier import logger

def normalize_splits(splits):
    mean = np.reshape(np.mean(splits["train"][0], axis = 1), (-1,1))
    std = np.reshape(np.std(splits["train"][0], axis = 1), (-1, 1))
    def _normalize(X, mean, std):
        return (X - mean)/std
    splits_norm = {k: (_normalize(v[0], mean, std), v[1], v[2]) for (k,v) in splits.items()}
    logger.info("Splits successfully normalized.")
    return splits_norm

def select_indices(n, p = 0.05):
    probabilities = np.random.rand(n)  # Generate n random numbers from U[0,1)
    selected_indices = np.where(probabilities < p)[0]  # Select indices where the random number is < 0.05
    return selected_indices

def translate_image(image, tx, ty):
    im  = image.reshape(32,32,3, order='F')
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(im, translation_matrix, (im.shape[1], im.shape[0]))
    return translated_image

def augment_data(X, Y, y):
    # randomly flipping with proba .05
    idx = select_indices(X.shape[1])
    X_added = np.zeros((3072, len(idx)))
    for i, ind in enumerate(idx):
        im  = X[:,ind].reshape(32,32,3, order='F')
        sim = np.flip(im, axis=0)
        X_added[:, i] = sim.flatten(order='F')
    y_added = y[idx]
    Y_added = Y[:, idx]
    X_aug = np.hstack((X, X_added))
    y_aug = np.concatenate((y, y_added))
    Y_aug = np.hstack((Y, Y_added))

    # randomly translating images with proba .05
    idx = select_indices(X_aug.shape[1])
    X_added = np.zeros((3072, len(idx)))
    for i, ind in enumerate(idx):
        sim = translate_image(X_aug[:,ind], np.random.randint(-10, 11), np.random.randint(-10, 11))
        X_added[:, i] = sim.flatten(order='F')
    y_added = y_aug[idx]
    Y_added = Y_aug[:, idx]
    X_aug = np.hstack((X_aug, X_added))
    y_aug = np.concatenate((y_aug, y_added))
    Y_aug = np.hstack((Y_aug, Y_added))
    return X_aug, Y_aug, y_aug

if __name__ == '__main__':
    fd = np.load('./data/full_data.npz')
    X_train, Y_train, y_train = fd["X_train"], fd["Y_train"], fd["y_train"]
    X_val, Y_val, y_val = fd["X_val"], fd["Y_val"], fd["y_val"]
    X_test, Y_test, y_test = fd["X_test"], fd["Y_test"], fd["y_test"]

    X_aug, Y_aug, y_aug = augment_data(X_train, Y_train, y_train)
    np.savez_compressed(
        "data/full_data_augmented", 
        X_train=X_aug, 
        X_val=X_val,
        Y_train=Y_aug,
        Y_val=Y_val,
        y_train=y_aug,
        y_val=y_val,
        X_test=X_test,
        Y_test=Y_test,
        y_test=y_test
    )