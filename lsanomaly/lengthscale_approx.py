# The MIT License (MIT
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
import logging

import numpy as np
from sklearn import neighbors

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def median_kneighbour_distance(X, k=5, seed=None):
    """
    Calculate the median distance between a set of random data points and
    their kth nearest neighbours. This is a heuristic for setting the
    kernel length scale.

    Args:
        X (numpy.ndarray): Data points
        k (int): Number of neighbors to use
        seed (int): random number seed

    Returns:
        float: Kernel length scale estimate

    Raises:
        ValueError: If the number of requested neighbors *k* is less than
        the number of observations in *X*.

    """
    if X.shape[0] < k:
        msg = "KNN cannot be run since k = {} is greater than the number of observations ({})".format(  # noqa
            k, X.shape[0]
        )
        raise ValueError(msg)

    if seed is not None:
        np.random.seed(seed)

    n_all = X.shape[0]
    n_subset = min(n_all, 2000)
    sample_idx_train = np.random.permutation(n_all)[:n_subset]

    nn = neighbors.NearestNeighbors(k)
    nn.fit(X[sample_idx_train, :])
    d, idx = nn.kneighbors(X[sample_idx_train, :])

    return np.median(d[:, -1])


def pair_distance_centile(X, centile, max_pairs=5000, seed=None):
    """
    Calculate centiles of distances between random pairs in a data-set.
    This an alternative to the median kNN distance for setting the kernel
    length scale.

    Args:
        X (numpy.ndarray): Data observations
        centile (int): distance centile
        max_pairs (int): maximum number of pairs to consider
        seed (int): random number seed

    Returns:
        float: length scale estimate
    """
    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]
    n_pairs = min(max_pairs, n ** 2)

    dists = np.zeros(n_pairs)

    for i in range(n_pairs):
        pair = np.random.randint(0, n, 2)
        pair_diff = X[pair[0], :] - X[pair[1], :]
        dists[i] = np.dot(pair_diff, pair_diff.T)
    dists.sort()

    out = dists[int(n_pairs * centile / 100.0)]
    return np.sqrt(out)
