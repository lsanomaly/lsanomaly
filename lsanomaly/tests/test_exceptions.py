import logging
import pytest

from lsanomaly import lengthscale_approx as lsa

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_exception_knn(example_arrays):
    X_train, X_test, _, _ = example_arrays
    with pytest.raises(ValueError):
        lsa.median_kneighbour_distance(X_test, k=5)


def test_exception_1d(example_arrays, anomaly_model):
    X_train, _, _, _ = example_arrays
    with pytest.raises(ValueError):
        anomaly_model.fit(X_train.ravel())


def test_exception_score(anomaly_model, example_arrays):
    X_train, _, _, _ = example_arrays
    with pytest.raises(ValueError):
        anomaly_model.score(X_train, None)


def test_exception_get(anomaly_model):
    with pytest.raises(NotImplementedError):
        anomaly_model.get_params()


def test_exception_set(anomaly_model):
    with pytest.raises(NotImplementedError):
        anomaly_model.set_params(x=1)


def test_exception_zero_ls(mc_model, pathological_x):
    with pytest.raises(ValueError):
        mc_model.fit(pathological_x)
