"""
Least squares anomaly detection demo on static data.

In this example, we generate some points from a 2-D Gaussian mixture,
and then plot the response of the anomaly detection model across the data
space given some different parameter settings.

The plots show training data as black crosses, contours in blue indicate
the response of the model across the space after training, and the contour
line in red indicates the decision boundary given by thresholding the model
output at 0.5.

This example was created by modifying the scikit-learn demo at
http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html
In general the LSAnomaly class can be used as a plug-in replacement in
any of the outlier detection demos on the sklearn site (e.g. as a
replacement for svm.OneClassSVM).
"""

import numpy as np
import pylab as plt

import lsanomaly

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def data_prep(n_samples=20, offset=2.5):
    xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))

    # Generate training data from a 2-D mixture model with two Gaussian
    # components
    X1 = np.random.randn(int(0.5 * n_samples), 2) - offset
    X2 = np.random.randn(int(0.5 * n_samples), 2) + offset
    X = np.r_[X1, X2]
    return X, xx, yy


def plot_results(
    X, xx, yy, threshold=0.5, sigma_candidates=None, rho_candidates=None
):
    _ = plt.figure(figsize=(16, 10))

    for row, sigma in enumerate(sigma_candidates):
        for col, rho in enumerate(rho_candidates):

            # Train the anomaly model
            clf = lsanomaly.LSAnomaly(sigma=sigma, rho=rho)
            clf.fit(X)

            # Get anomaly scores across the grid
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot the training data, anomaly model response and decision
            # boundary at threshold 0.5.
            subplot = plt.subplot(
                len(sigma_candidates), len(rho_candidates), row * 3 + col + 1
            )
            plt.contourf(
                xx,
                yy,
                Z,
                levels=np.linspace(0, 1, 11),
                cmap=plt.cm.get_cmap("GnBu"),
            )
            subplot.contour(
                xx, yy, Z, levels=[threshold], linewidths=2, colors="red"
            )
            cb = plt.colorbar()
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(10)
            plt.scatter(
                X[:, 0], X[:, 1], c="black", marker="+", s=50, linewidth=2
            )
            subplot.set_title(
                "$\sigma = $ %.3g, $\\rho$ = %.3g" % (sigma, rho),
                fontsize=14,
                usetex=True,
            )
            subplot.axes.get_xaxis().set_ticks([])
            subplot.axes.get_yaxis().set_ticks([])

            plt.xlim((-7, 7))
            plt.ylim((-7, 7))

    plt.show()
