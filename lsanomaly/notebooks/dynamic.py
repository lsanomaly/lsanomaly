"""
Least squares anomaly detection in sequences.

Example of detecting periods of abnormality in a time series of physiological
measurements.
"""
import json

import numpy as np
import pylab as plt


def data_prep(data_file="filtered_ecg.json", lag=10):
    with open(data_file) as f:
        X = np.array(json.load(f))

    # Construct 4-D  observations from the original 2-D data: values at the
    # current index and at a fixed lag behind.
    N = X.shape[0]
    X2 = np.zeros((N - lag, 4))
    for i in range(lag, N):
        X2[i - lag, 0] = X[i, 0]
        X2[i - lag, 1] = X[i - lag, 0]
        X2[i - lag, 2] = X[i, 1]
        X2[i - lag, 3] = X[i - lag, 1]

    X_train = X2[:5000, :]
    X_test = X2[10000:15000, :]
    return X_train, X_test


def plot_results(X_test, y_pred_static, y_pred_dynamic, static_threshold=0.0):
    _ = plt.figure(figsize=(12, 6))
    f_size = 16

    plt.subplot(4, 1, 1)
    plt.plot(X_test[:, 1])
    plt.ylabel("ECG 1", rotation="horizontal", fontsize=f_size, ha="right")
    plt.grid(which="major", axis="x")
    plt.xticks([], "", fontsize=f_size)

    plt.title(
        "Detection of cardiac arrhythmia from ECG sequence",
        fontsize=f_size + 2,
        fontweight="medium",
    )
    plt.subplot(4, 1, 2)
    plt.plot(X_test[:, 3])
    plt.xticks([])
    plt.ylabel("ECG 2", rotation="horizontal", fontsize=f_size, ha="right")

    # static scores
    plt.subplot(4, 1, 3)
    plt.plot(y_pred_static[:, 1], "r")
    plt.xticks([])
    plt.ylim([-0.05, 1.05])
    plt.ylabel(
        "\nAnomaly\nscore\n(static)",
        rotation="horizontal",
        ha="right",
        fontsize=f_size,
    )

    # dynamic scores
    plt.subplot(4, 1, 4)
    plt.plot(y_pred_dynamic[:, 1], "r")
    plt.ylim([-0.05, 1.05])
    plt.ylabel(
        "\nAnomaly\nscore\n(dynamic)",
        rotation="horizontal",
        ha="right",
        fontsize=f_size,
    )
    return plt.show()
