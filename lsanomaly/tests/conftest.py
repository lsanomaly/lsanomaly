import json
import logging
import os

import numpy as np
import pytest

from lsanomaly import LSAnomaly

log_fmt = "[%(asctime)s %(levelname)-8s], [%(filename)s:%(lineno)s - %(funcName)s()], %(message)s"  # noqa
logging.basicConfig(level=logging.DEBUG, format=log_fmt)

here = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(here, "data")


def load_json(f_name):
    with open(os.path.join(test_data_dir, f_name)) as f:
        data = json.load(f)
        return np.array(data)


@pytest.fixture(scope="session")
def seed():
    return 42


@pytest.fixture(scope="session")
def check_ndarray():
    def check(test, expected):
        assert np.allclose(test, expected, rtol=10e-5, atol=10e-5)

    return check


@pytest.fixture(scope="session")
def example_arrays():
    X_train = np.array([[1.1], [1.3], [1.2], [1.05], [1.2]])
    X_test = np.array([[1.15], [3.6], [1.25]])
    expected_predict = [0.0, 1.0, 0.0]
    expected_prob = np.array(
        [
            [1.00000000e00, 0.00000000e00],
            [1.66850204e-58, 1.00000000e00],
            [9.93806250e-01, 6.19375049e-03],
        ]
    )
    return X_train, X_test, expected_predict, expected_prob


@pytest.fixture(scope="session")
def doc_arrays():
    x_train = np.array([[1], [2], [3], [1], [2], [3]])
    predict_prob = np.array([[0.7231233, 0.2768767]])
    return x_train, predict_prob


@pytest.fixture(scope="session")
def anomaly_model():
    return LSAnomaly(rho=1, sigma=0.5, seed=42)


@pytest.fixture(scope="function")
def mc_model():
    return LSAnomaly(seed=42)


@pytest.fixture(scope="session")
def x_train_test_ecg():
    x_train = load_json("x_train_ecg.json")
    x_test = load_json("x_test_ecg.json")
    return x_train, x_test


@pytest.fixture(scope="session")
def y_expected_static_ecg():
    y_pred_static = load_json("y_pred_ecg_static.json")
    return y_pred_static


@pytest.fixture(scope="session")
def y_expected_dynamic_ecg():
    y_pred_dyn = load_json("y_pred_ecg_dynamic.json")
    return y_pred_dyn


@pytest.fixture(scope="session")
def y_expected_seq_smoothing():
    y_pred_smooth = load_json("y_pred_ecg_seq_smoothing.json")
    return y_pred_smooth


@pytest.fixture(scope="session")
def y_expected_seq_filtering():
    y_pred_smooth = load_json("y_pred_ecg_seq_filtering.json")
    return y_pred_smooth


@pytest.fixture(scope="session")
def sequence_params():
    A = np.array([[0.999, 0.001], [0.01, 0.99]])
    pi = np.array([0.5, 0.5])
    return A, pi


@pytest.fixture(scope="session")
def digits_x_y():
    X_train = load_json("digits_x_train.json")
    X_test = load_json("digits_x_test.json")
    y_train = load_json("digits_y_train.json")
    y_test = load_json("digits_y_test.json")
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def multiclass1_digits_expected():
    y_pred = load_json("mc_digits_predictions.json")
    return y_pred


@pytest.fixture(scope="session")
def multiclass1_digits_scores():
    train_score = 0.99012
    test_score = 0.86207
    return train_score, test_score


@pytest.fixture(scope="session")
def multiclass1_digits_inlier_scores():
    in_s = load_json("mc_digits_inlier_score.json")
    return in_s


@pytest.fixture(scope="session")
def check_ps():
    def check_p(p):
        loc = np.where(np.logical_and(p < 0.0, p > 1.0))
        assert loc[0].size == 0

    return check_p
