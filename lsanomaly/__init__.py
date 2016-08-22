"""Least squares anomaly detection."""

__title__ = 'lsanomaly'
__version__ = '0.1.3'
__license__ = 'MIT'

import numpy as np
import copy
from sklearn import metrics, base, neighbors


def median_kneighbour_distance(X, k=5):
    """
    Calculate the median kneighbor distance.

    Find the distance between a set of random datapoints and
    their kth nearest neighbours. This is a heuristic for setting the
    kernel length scale.
    """
    N_all = X.shape[0]
    N_subset = min(N_all, 2000)
    sample_idx_train = np.random.permutation(N_all)[:N_subset]
    nn = neighbors.NearestNeighbors(k)
    nn.fit(X[sample_idx_train, :])
    d, idx = nn.kneighbors(X[sample_idx_train, :])
    return np.median(d[:, -1])


def pair_distance_centile(X, centile, max_pairs=5000):
    """
    Calculate centiles of distances between random pairs in a dataset.

    This an alternative to the median kNN distance for setting the kernel
    length scale.
    """
    N = X.shape[0]
    n_pairs = min(max_pairs, N**2)
    # randorder1 = np.random.permutation(N)
    # randorder2 = np.random.permutation(N)

    dists = np.zeros(n_pairs)

    for i in range(n_pairs):
        pair = np.random.randint(0, N, 2)
        pairdiff = X[pair[0], :]-X[pair[1], :]
        dists[i] = np.dot(pairdiff, pairdiff.T)
    dists.sort()

    out = dists[int(n_pairs*centile/100.)]
    return np.sqrt(out)


class LSAnomaly(base.BaseEstimator):
    """
    Class for training an inlier model and predicting outliers.

    Probabilities for test data, using a least-squares kernel-based method.

    Parameters
    ----------
    n_kernels_max : int, optional (default 500)
        Maximum number of kernel basis centres to use for modelling inlier
        classes.
    kernel_pos : optional
        The positions of the kernel centres can be specified. Default is to
        select them as a random subset of the training data.
    sigma : float, optional
        Kernel scale parameter. If set to 'mediandist', then at training time
        half the median distance between a random set of pairs of datapoints
        in the training data will be used as a default setting. This is
        usually within an order of magnitude of the optimum setting.
    gamma : float, optional
        An alternative way of specifying the kernel scale parameter,
        for compatibility with SVM notation. sigma = 1/sqrt(gamma)
    rho : float, optional
        Regularization parameter (higher values give greater sensitivity to
        outliers).

    Example
    -------
    >>> import lsanomaly
    >>> X_train = np.array([[1.1],[1.3],[1.2],[1.05]])
    >>> X_test = np.array([[1.15],[3.6],[1.25]])
    >>> anomalymodel.fit(X_train)
    >>> anomalymodel.predict(X_test)
    [0.0, 1.0, 0.0]
    >>> anomalymodel.predict_proba(X_test)
    array([[  1.00000000e+000,   0.00000000e+000],
           [  5.15255628e-103,   1.00000000e+000],
           [  1.00000000e+000,   0.00000000e+000]])
    """

    def __init__(self, n_kernels_max=500, kernel_pos=None, sigma=None,
                 rho=None, gamma=None):
        """Initialize a new estimator instance with provided parameters."""
        self.n_kernels_max = n_kernels_max
        self.kernel_pos = kernel_pos
        self.sigma = sigma
        self.gamma = gamma
        self.rho = rho
        self.theta = None
        if not (sigma is None or isinstance(sigma, str)):
            self.gamma = sigma**-2
        if gamma is not None:
            self.sigma = gamma**-.5

    def fit(self, X, y=None):
        """
        Fit the inlier model given training data.

        This function attempts to choose reasonable defaults for parameters
        sigma and rho if none are specified, which could then be adjusted
        to improve performance.

        Parameters
        ----------
        X : array
            Examples of inlier data, of dimension N times d (rows are
            examples, columns are data dimensions)
        y : array, optional
            If the inliers have multiple classes, then y contains the class
            assignments as a vector of length N. If this is specified then
            the model will attempt to assign test data to one of the inlier
            classes or to the outlier class.
        """
        N = X.shape[0]

        if y is None:
            y = np.zeros(N)

        self.classes = list(set(y))
        self.classes.sort()
        self.n_classes = len(self.classes)

        # If no kernel parameters specified, try to choose some defaults
        if not self.sigma:
            self.sigma = median_kneighbour_distance(X)
            self.gamma = self.sigma**-2

        if not self.gamma:
            self.gamma = self.sigma**-2

        if not self.rho:
            self.rho = 0.1

        # choose kernel basis centres
        if self.kernel_pos is None:
            B = min(self.n_kernels_max, N)
            kernel_idx = np.random.permutation(N)
            self.kernel_pos = X[kernel_idx[:B]]
        else:
            B = self.kernel_pos.shape[0]

        # fit coefficients
        Phi = metrics.pairwise.rbf_kernel(X, self.kernel_pos, self.gamma)
        theta = {}
        Phi_PhiT = np.dot(Phi.T, Phi)
        inverse_term = np.linalg.inv(Phi_PhiT + self.rho*np.eye(B))
        for c in self.classes:
            m = (y == c).astype(int)
            theta[c] = np.dot(inverse_term, np.dot(Phi.T, m))

        self.theta = theta

    def predict(self, X):
        """
        Assign classes to test data.

        Parameters
        ----------
        X : array
            Test data, of dimension N times d (rows are examples, columns
            are data dimensions)

        Returns
        -------
        y_predicted : array
            A vector of length N containing assigned classes. If no inlier
            classes were specified during training, then 0 denotes an inlier
            and 1 denotes an outlier. If multiple inlier classes were
            specified, then each element of y_predicted is either on of
            those inlier classes, or an outlier class (denoted by the
            maximum inlier class ID plus 1).
        """
        predictions_proba = self.predict_proba(X)
        predictions = []
        allclasses = copy.copy(self.classes)
        allclasses.append('anomaly')
        for i in range(X.shape[0]):
            predictions.append(allclasses[predictions_proba[i, :].argmax()])
        return predictions

    def predict_proba(self, X):
        """
        Calculate posterior probabilities of test data.

        Parameters
        ----------
        X : array
            Test data, of dimension N times d (rows are examples, columns
            are data dimensions)

        Returns:
        -------
        y_prob : array
            An array of dimension N times n_inlier_classes+1, containing
            the probabilities of each row of X being one of the inlier
            classes, or the outlier class (last column).
        """
        Phi = metrics.pairwise.rbf_kernel(X, self.kernel_pos, self.gamma)
        N = X.shape[0]
        predictions = np.zeros((N, self.n_classes+1))
        for i in range(N):
            post = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                post[c] = max(0,
                              np.dot(self.theta[self.classes[c]].T, Phi[i, :]))
                post[c] = min(post[c], 1.)
            predictions[i, :-1] = post
            predictions[i, -1] = max(0, 1-sum(post))

        return predictions

    def decision_function(self, X):
        """
        Generate an inlier score for each test data example.

        Parameters
        ----------
        X : array
            Test data, of dimension N times d (rows are examples, columns
            are data dimensions)

        Returns:
        -------
        scores : array
            A vector of length N, where each element contains an inlier
            score in the range 0-1 (outliers have values close to zero,
            inliers have values close to one).
        """
        predictions = self.predict_proba(X)
        out = np.zeros((predictions.shape[0], 1))
        out[:, 0] = 1 - predictions[:, -1]
        return out

    def score(self, X, y):
        """
        Calculate accuracy score.

        Needed because of bug in metrics.accuracy_score when comparing
        list with numpy array.
        """
        predictions = self.predict(X)
        true = 0.0
        total = 0.0
        for i in range(len(predictions)):
            total += 1
            if predictions[i] == y[i]:
                true += 1
        return true/total

    def predict_sequence(self, X, A, pi, inference='smoothing'):
        """
        Calculate class probabilities for a sequence of data.

        Parameters
        ----------
        X : array
            Test data, of dimension N times d (rows are time frames, columns
            are data dimensions)
        A : class transition matrix, where A[i,j] contains p(y_t=j|y_{t-1}=i)
        pi : vector of initial class probabilities
        inference : can be 'smoothing' or 'filtering'.

        Returns:
        -------
        y_prob : array
            An array of dimension N times n_inlier_classes+1, containing
            the probabilities of each row of X being one of the inlier
            classes, or the outlier class (last column).
        """
        obsll = self.predict_proba(X)
        T, S = obsll.shape
        alpha = np.zeros((T, S))

        alpha[0, :] = pi
        for t in range(1, T):
            alpha[t, :] = np.dot(alpha[t-1, :], A)
            for s in range(S):
                alpha[t, s] *= obsll[t, s]
            alpha[t, :] = alpha[t, :]/sum(alpha[t, :])

        if inference == 'filtering':
            return alpha
        else:
            beta = np.zeros((T, S))
            gamma = np.zeros((T, S))
            beta[T-1, :] = np.ones(S)
            for t in range(T-2, -1, -1):
                for i in range(S):
                    for j in range(S):
                        beta[t, i] += A[i, j]*obsll[t+1, j]*beta[t+1, j]
                beta[t, :] = beta[t, :]/sum(beta[t, :])

            for t in range(T):
                gamma[t, :] = alpha[t, :]*beta[t, :]
                gamma[t, :] = gamma[t, :]/sum(gamma[t, :])

            return gamma
