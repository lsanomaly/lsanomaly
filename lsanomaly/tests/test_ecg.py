import logging
import pytest


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_ecg_static(
    anomaly_model, x_train_test_ecg, y_expected_static_ecg, check_ndarray
):
    logger.debug(anomaly_model)
    x_train, x_test = x_train_test_ecg

    anomaly_model.fit(x_train)

    y_pred_static = anomaly_model.predict_proba(x_test)
    check_ndarray(y_expected_static_ecg, y_pred_static)


def test_ecg_dynamic(
    anomaly_model, x_train_test_ecg, y_expected_dynamic_ecg, check_ndarray
):
    logger.debug(anomaly_model)
    x_train, x_test = x_train_test_ecg
    anomaly_model.fit(x_train)

    y_pred_static = anomaly_model.predict_proba(x_test)
    check_ndarray(y_expected_dynamic_ecg, y_pred_static)


@pytest.mark.parametrize("inference", ["smoothing", "filtering"])
def test_ecg_sequence(
    inference,
    anomaly_model,
    x_train_test_ecg,
    sequence_params,
    y_expected_seq_smoothing,
    y_expected_seq_filtering,
    check_ndarray,
):
    x_train, x_test = x_train_test_ecg
    A, pi = sequence_params

    anomaly_model.fit(x_train)
    y_pred_seq = anomaly_model.predict_sequence(
        x_test, A, pi, inference=inference
    )
    if inference == "smoothing":
        check_ndarray(y_pred_seq, y_expected_seq_smoothing)
    elif inference == "filtering":
        check_ndarray(y_pred_seq, y_expected_seq_filtering)
