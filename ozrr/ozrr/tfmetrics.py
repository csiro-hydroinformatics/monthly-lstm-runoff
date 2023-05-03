"""
A module of metrics to support hydrology
"""

from typing import Tuple, Union

import tensorflow as tf
keras = tf.keras
from keras.metrics import mean_absolute_error, mean_absolute_percentage_error

import numpy as np

NdArray = Union[tf.Tensor, np.ndarray]

# Type hints are probably not quite correct: numpy or tensors is what is probably OK. Not readily documented in Keras.


# def custom_error_function(y_true:NdArray, y_pred:NdArray) -> float:
#     import tensorflow.keras.backend as K
#     bool_finite = tf.is_finite(y_true)
#     return K.mean(K.square(tf.boolean_mask(y_pred, bool_finite) - tf.boolean_mask(y_true, bool_finite)), axis=-1)

def remove_items_missing_observations(y_true:NdArray, y_pred:NdArray) -> Tuple[NdArray,NdArray]:
    """Removes from both vectors items where y_true is not finite

    Args:
        y_true (np.ndarray): observations
        y_pred (np.ndarray): predictions

    Returns:
        Tuple[np.ndarray,np.ndarray]: observations and predictions with items where observations were missing removed from both vectors
    """
    if y_true.shape != y_pred.shape:
        print(f"y_true = {y_true}")
        print(f"y_pred = {y_pred}")
        raise ValueError(f"y_true and y_pred do not have compatible shapes: {y_true.shape} and {y_pred.shape}. Respective types are {type(y_true)} and {type(y_pred)}")
    # import tensorflow.keras.backend as K
    bool_finite = tf.math.is_finite(y_true)
    y_pred_m = tf.boolean_mask(y_pred, bool_finite)
    y_true_m = tf.boolean_mask(y_true, bool_finite)
    return y_true_m, y_pred_m


def nse_loss(y_true:NdArray, y_pred:NdArray) -> float:
    y_true_m, y_pred_m = remove_items_missing_observations(y_true, y_pred)
    denominator = tf.reduce_sum(tf.square(y_true_m - tf.reduce_mean(y_true_m)))
    numerator = tf.reduce_sum(tf.square(y_pred_m - y_true_m))
    nse_val = tf.divide(numerator, denominator)
    return nse_val

def nse(y_true:NdArray, y_pred:NdArray) -> float:
    """Nash-Sutcliffe efficiency on items where observations are not missing"""
    y_loss = nse_loss(y_true, y_pred)
    nse_val = tf.subtract(
        tf.constant(1, dtype=tf.float32), y_loss
    )
    return nse_val

def mean_absolute_error_na(y_true:NdArray, y_pred:NdArray) -> float:
    """MAE on items where observations are not missing"""
    y_true_m, y_pred_m = remove_items_missing_observations(y_true, y_pred)
    return mean_absolute_error(y_true_m, y_pred_m)

def mean_absolute_percentage_error_na(y_true:NdArray, y_pred:NdArray) -> float:
    """Percentage MAE on items where observations are not missing"""
    y_true_m, y_pred_m = remove_items_missing_observations(y_true, y_pred)
    return mean_absolute_percentage_error(y_true_m, y_pred_m)

def peak_loss(y_true:NdArray, y_pred:NdArray) -> float:
    raise NotImplementedError("peak_loss is not documented nor unit tested")
    # y_true_m, y_pred_m = remove_items_missing_observations(y_true, y_pred)
    # mse = mean_squared_error(y_true_m, y_pred_m)
    # peak_num = tf.square(tf.reduce_max(y_pred_m) - tf.reduce_max(y_true_m))
    # peak_denom = tf.square(tf.reduce_max(y_true_m))
    # peak = tf.divide(peak_num, peak_denom)
    # return tf.add(mse, peak)


def ymb_loss(y_true:NdArray, y_pred:NdArray) -> float:
    raise NotImplementedError("ymb_loss is not documented nor unit tested")
    # y_true_m, y_pred_m = remove_items_missing_observations(y_true, y_pred)
    # diff_mass_bal = tf.divide(
    #     tf.abs(tf.reduce_sum(y_true_m) - tf.reduce_sum(y_pred_m)),
    #     tf.reduce_sum(y_true_m),
    # )
    # return tf.add(diff_mass_bal, mean_squared_error(y_true_m, y_pred_m))
