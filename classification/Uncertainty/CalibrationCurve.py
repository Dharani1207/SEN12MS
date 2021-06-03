import warnings

from sklearn.calibration import calibration_curve
import numpy as np


def get_calibration_curve(y_true, y_pred, strategy='uniform', n_bins=10):

    assert np.shape(y_true)[-1] == 1, "input data should be binary. use 'get_multiclass_calibration_curve' instead!"

    prob_true, prob_pred = calibration_curve(y_true, y_pred, strategy=strategy, n_bins=n_bins)

    return prob_true, prob_pred


def get_multiclass_calibration_curve(y_true, y_pred, strategy='uniform', n_bins=10):
    confidence = np.max(y_pred, axis=-1, keepdims=True)
    true_positive = np.expand_dims((np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)).astype('Int32'), axis=-1)

    return get_calibration_curve(true_positive, confidence)


if __name__=="__main__":

    y_pred = [[0.4,0.6],[0.9,0.1],[0.3,0.7]]
    y_true = [[1.0,0.0],[1.0,0.0],[0.0,1.0]]

    a, b = get_multiclass_calibration_curve(y_true, y_pred)
    print(a)
    print(b)