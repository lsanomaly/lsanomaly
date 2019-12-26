"""
Demo of least squares anomaly detection on static digits data.

In this example, we try to recognise digits of class 9 given training
examples from classes 0-8.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, model_selection


def plot_roc(fper, tper, auc):
    _ = plt.figure(figsize=(8, 6))

    plt.plot(fper, tper, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "Receiver Operating Characteristic (ROC) Curve\nAUC = {:1.3f}".format(  # noqa
        auc
    )
    plt.title(title, fontsize=16)
    plt.legend()
    plt.show()


def data_prep(test_size=0.2):
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # Split data into training and test sets, then remove all examples of
    # class 9 from the training set, leaving only examples of 0-8.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size
    )

    train_inlier_idx = y_train < 9
    X_train = X_train[train_inlier_idx, :]
    y_train = y_train[train_inlier_idx]
    return X_train, X_test, y_train, y_test


# adapted from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(
    y_true,
    y_pred,
    target_names=None,
    title="Confusion matrix",
    cmap=None,
    normalize=True,
):
    cm = metrics.confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=14)
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
            accuracy, misclass
        ),
        fontsize=14,
    )
    plt.show()
