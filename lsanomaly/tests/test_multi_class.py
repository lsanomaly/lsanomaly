import logging
import numpy as np


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_multiclass_digits_prediction(
    mc_model, digits_x_y, multiclass1_digits_expected, check_ndarray, check_ps
):
    X_train, X_test, y_train, y_test = digits_x_y

    logger.debug(mc_model)
    mc_model.fit(X_train, y_train)

    predictions = mc_model.predict_proba(X_test)
    check_ps(predictions)
    check_ndarray(predictions, multiclass1_digits_expected)


def test_multiclass_digits_score(
    mc_model, digits_x_y, multiclass1_digits_scores
):
    X_train, X_test, y_train, y_test = digits_x_y
    train_score, test_score = multiclass1_digits_scores

    mc_model.fit(X_train, y_train)

    s = mc_model.score(X_train, y_train)
    logger.debug("score: {:6.4f}".format(s))
    assert np.isclose(s, train_score)

    s = mc_model.score(X_test, y_test)
    logger.debug("score: {:6.4f}".format(s))
    assert np.isclose(s, test_score)


def test_multiclass_digits_df(
    mc_model,
    digits_x_y,
    multiclass1_digits_inlier_scores,
    check_ndarray,
    check_ps,
):
    X_train, X_test, y_train, _ = digits_x_y

    mc_model.fit(X_train, y_train)

    inlier_score = mc_model.decision_function(X_test)

    assert inlier_score.shape[0] == X_test.shape[0]
    check_ps(inlier_score)
    check_ndarray(inlier_score, multiclass1_digits_inlier_scores)
