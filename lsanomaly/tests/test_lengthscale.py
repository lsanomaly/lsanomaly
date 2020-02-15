import logging
import numpy as np

from lsanomaly.lengthscale_approx import median_kneighbour_distance


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_ls_knn(example_arrays, seed):
    x_train, _, _, _ = example_arrays
    ls_approx = median_kneighbour_distance(x_train, k=5, seed=seed)
    assert np.isclose(ls_approx, 0.20)
