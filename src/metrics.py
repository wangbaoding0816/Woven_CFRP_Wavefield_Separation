import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def v_theta(vx, vy):
    """
    Compute angular velocity component v(theta) from vector field components.
    vx, vy: arrays of the same shape.
    """
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    return np.arctan2(vy, vx)
