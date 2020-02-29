"""
Least Squares Anomaly Detection

"""
# The MIT License (MIT)
#
# Copyright (c) 2016 John Quinn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal i n the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
import copy
import logging
import time

import numpy as np
from sklearn import metrics, base

import lsanomaly.version as v
from lsanomaly.lengthscale_approx import median_kneighbour_distance

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _check_shape(x):
    if x.ndim < 2:
        msg = "array must have dimension > 1"
        logger.error(msg)
        raise ValueError(msg)
    else:
        logger.debug("array shape: {}".format(x.shape))


def _check_sigma(sigma):
    if np.isclose(np.array(sigma), np.array([0.0])):
        raise ValueError(
            "knn distance is too small, got {:0.4f}".format(sigma)
        )  # noqa


class LSAnomaly(base.BaseEstimator):
    __version__ = v.__version__

    def __init__(
        self,
        n_kernels_max=500,
        kernel_pos=None,
        sigma=None,
        rho=None,
        gamma=None,
        seed=None,
    ):
        """
        Class for training an inlier model and predicting outlier
        probabilities for test data, using a least-squares kernel-based method.

        Args:
            n_kernels_max (int): optional (default 500)

            kernel_pos (numpy.ndarray): The positions of the kernel centers can
            be specified. Default is to select them as a random subset of the
            training data.

            sigma (float|str): Kernel length scale parameter. If set to
            'mediandist', then at training time half the median distance
            between a random set of pairs of data points in the training data
            will be used as a default setting. See
            :py:meth: `lengthscale_ approx`

            rho (float): Regularization parameter. Higher values give greater
            sensitivity to outliers.

            gamma (float): An alternative way of specifying the kernel
            length scale parameter, for compatibility with SVM notation:
            `sigma = 1/sqrt(gamma)`.

            seed (int): Random number seed

        Example
            >>> from lsanomaly import LSAnomaly
            >>> anomaly_model = LSAnomaly()
            >>> X_train = np.array([[1.1], [1.3], [1.2], [1.05], [1.2]])
            >>> X_test = np.array([[1.15], [3.6], [1.25]])
            >>> anomaly_model.fit(X_train)
            >>> anomaly_model.predict(X_test)
            [0.0, 1.0, 0.0]
            >>> anomaly_model.predict_proba(X_test)
            array[[1.00000000e+00 0.00000000e+00]
            [1.66850204e-58 1.00000000e+00]
            [9.93806250e-01 6.19375049e-03]]
        """
        self.n_kernels_max = n_kernels_max
        self.kernel_pos = kernel_pos
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.seed = seed

        self.theta = dict()
        self.classes = None
        self.n_classes = None

        if not (sigma is None or isinstance(sigma, str)):
            _check_sigma(sigma)
            self.gamma = sigma ** -2
        if gamma is not None:
            self.sigma = gamma ** -0.5

        self.supported_inferences = ["smoothing", "filtering"]

        if seed is not None:
            np.random.seed(self.seed)

    def fit(self, X, y=None, k=5):
        """
        Fit the inlier model given training data. This function attempts
        to choose reasonable defaults for parameters sigma and rho if none
        are specified, which could then be adjusted to improve performance.

        Args:
            X (numpy.ndarray): Examples of inlier data, of dimension N
            times d (rows are examples, columns are data dimensions)

            y (numpy.ndarray): If the inliers have multiple classes, then `y`
            contains the class assignments as a vector of length N. If this is
            specified then the model will attempt to assign test data to one
            of the inlier classes or to the outlier class.

            k (int): Number of nearest neighbors to use in the KNN
            kernel length scale heuristic.

        Returns:
            self

        """
        start = time.time()
        _check_shape(X)

        N = X.shape[0]
        logger.debug("X shape: {}".format(X.shape))

        if not isinstance(y, np.ndarray):
            y = np.zeros(N)

        self.classes = list(set(y))
        self.classes.sort()
        self.n_classes = len(self.classes)

        logger.debug("number of classes: {}".format(self.n_classes))

        # If no kernel parameters specified, try to choose some defaults
        if not self.sigma:
            self.sigma = median_kneighbour_distance(X, k=k)
            _check_sigma(self.sigma)
            self.gamma = self.sigma ** -2

        if not self.gamma:
            self.gamma = self.sigma ** -2

        if not self.rho:
            self.rho = 0.1

        logger.debug("sigma : {:>6.4f}".format(self.sigma))
        logger.debug("gamma : {:>6.4f}".format(self.gamma))
        logger.debug("rho   : {:>6.4f}".format(self.rho))

        # choose kernel basis centres
        if self.kernel_pos is None:
            B = min(self.n_kernels_max, N)
            kernel_idx = np.random.permutation(N)
            self.kernel_pos = X[kernel_idx[:B]]
        else:
            B = self.kernel_pos.shape[0]

        # fit coefficients
        phi = metrics.pairwise.rbf_kernel(X, self.kernel_pos, self.gamma)
        phi_dot_phi = np.dot(phi.T, phi)
        inverse_term = np.linalg.inv(phi_dot_phi + self.rho * np.eye(B))
        # self._iter_classes(y, phi, inverse_term)
        for c in self.classes:
            m = (y == c).astype(int)
            self.theta[c] = np.dot(inverse_term, np.dot(phi.T, m))
        logger.debug("that took {:6.4f}s".format(time.time() - start))
        return self

    def get_params(self, deep=True):
        """
        Not implemented.

        Args:
            deep (bool):

        Returns:

        """
        raise NotImplementedError

    def set_params(self, **params):
        """
        Not implemented.

        Args:
            **params (dict):

        """
        raise NotImplementedError

    def predict(self, X):
        """
        Assign classes to test data.

        Args:
            X (numpy.ndarray): Test data, of dimension N times d (rows are
            examples, columns are data dimensions)

        Returns:
            numpy.ndarray:

            A vector of length N containing assigned classes.
            If no inlier classes were specified during training, then 0 denotes
            an inlier and 1 denotes an outlier. If multiple inlier classes were
            specified, then each element of y_predicted is either one of those
            inlier classes, or an outlier class (denoted by the maximum inlier
            class ID plus 1).

        """
        _check_shape(X)

        predictions_proba = self.predict_proba(X)
        all_classes = copy.copy(self.classes)
        all_classes.append(1.0)
        predictions = [
            all_classes[predictions_proba[i, :].argmax()]
            for i in range(X.shape[0])
        ]
        return predictions

    def predict_proba(self, X):
        """
        Calculate posterior probabilities of each inlier class and the
        outlier class for test data.

        Args
            X (numpy.ndarray): Test data, of dimension N times d (rows are
            examples, columns are data dimensions)

        Returns
            numpy.ndarray:
            An array of dimension N times n_inlier_classes+1,
            containing the probabilities of each row of X being one of the
            inlier classes, or the outlier class (last column).

        """
        _check_shape(X)

        phi = metrics.pairwise.rbf_kernel(X, self.kernel_pos, self.gamma)
        n = X.shape[0]
        predictions = np.zeros((n, self.n_classes + 1))
        for i in range(n):
            post = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                post[c] = max(
                    0.0,
                    float(np.dot(self.theta[self.classes[c]].T, phi[i, :])),
                )
                post[c] = min(post[c], 1.0)
            predictions[i, :-1] = post
            predictions[i, -1] = max(0, 1 - sum(post))

        return predictions

    def decision_function(self, X):
        """
        Generate an inlier score for each test data example.

        Args
            X (numpy.ndarray): Test data, of dimension N times d (rows are
            examples, columns are data dimensions)

        Returns:
            numpy.ndarray:
            A vector of length N, where each element contains an
            inlier score in the range 0-1 (outliers have values close to zero,
            inliers have values close to one).

        """
        _check_shape(X)
        predictions = self.predict_proba(X)
        out = np.zeros((predictions.shape[0], 1))
        out[:, 0] = 1 - predictions[:, -1]
        return out

    def score(self, X, y):
        """
        Calculate accuracy score, needed because of bug in
        metrics.accuracy_score when comparing list with numpy array.
        """
        _check_shape(X)
        if y is None:
            raise ValueError("y cannot be None")

        predictions = self.predict(X)
        total = len(predictions)
        tf = [int(predictions[i] == y[i]) for i in range(total)]
        return sum(tf) / total

    def predict_sequence(self, X, A, pi, inference="smoothing"):
        """
        Calculate class probabilities for a sequence of data.

        Args
            X (numpy.ndarray): Test data, of dimension N times d (rows are time
            frames, columns are data dimensions)

            A (numpy.ndarray):: Class transition matrix, where A[i,j]
            contains p(y_t=j|y_{t-1}=i)

            pi (numpy.ndarray): vector of initial class probabilities

            inference (str) : 'smoothing' or 'filtering'.

        Returns
            numpy.ndarray: An array of dimension N times n_inlier_classes+1,
            containing the probabilities of each row of X being one of the
            inlier classes, or the outlier class (last column).

        """
        _check_shape(X)
        if inference not in self.supported_inferences:
            logger.warning(
                "expecting one of {}; got {}".format(
                    self.supported_inferences, inference
                )
            )
            inference = "smoothing"
        logger.debug("using inference: {}", format(inference))

        obs_all = self.predict_proba(X)
        T, S = obs_all.shape
        alpha = np.zeros((T, S))

        alpha[0, :] = pi
        for t in range(1, T):
            alpha[t, :] = np.dot(alpha[t - 1, :], A)
            for s in range(S):
                alpha[t, s] *= obs_all[t, s]
            alpha[t, :] = alpha[t, :] / sum(alpha[t, :])

        if inference == "filtering":
            y_prob = alpha
        else:
            beta = np.zeros((T, S))
            gamma = np.zeros((T, S))
            beta[T - 1, :] = np.ones(S)
            for t in range(T - 2, -1, -1):
                for i in range(S):
                    for j in range(S):
                        beta[t, i] += (
                            A[i, j] * obs_all[t + 1, j] * beta[t + 1, j]
                        )
                beta[t, :] = beta[t, :] / sum(beta[t, :])

            for t in range(T):
                gamma[t, :] = alpha[t, :] * beta[t, :]
                gamma[t, :] = gamma[t, :] / sum(gamma[t, :])
            y_prob = gamma
        return y_prob
