import logging

import numpy as np
from lsanomaly import LSAnomaly

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_example_code(mc_model, example_arrays, check_ndarray, check_ps):
    x_train, x_test, expected_predict, expected_prob = example_arrays

    mc_model.fit(x_train)
    p = mc_model.predict(x_test)
    logger.debug("predict = {}".format(p))

    assert p == expected_predict

    p = mc_model.predict_proba(x_test)
    logger.debug("probs = {}".format(p))

    check_ps(p)
    check_ndarray(p, expected_prob)


def test_example_doc(doc_arrays, check_ndarray):
    test_pt = np.array([[0]])
    x_train, predict_prob = doc_arrays

    anomaly_model = LSAnomaly(sigma=3, rho=0.1, seed=42)
    anomaly_model.fit(x_train)

    expected = [0.0]
    p = anomaly_model.predict(test_pt)
    assert p == expected

    expected = np.array([[0.7231233, 0.2768767]])
    p = anomaly_model.predict_proba(test_pt)

    logger.debug("p = {}".format(p))
    check_ndarray(expected, p)
