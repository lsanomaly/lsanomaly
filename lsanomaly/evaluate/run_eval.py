"""
**run_eval.py**

Least squares anomaly evaluation on static data. After running experiments,
use `generate_latex.py` to create a table of results.

This is a refactored and updated version of
the script in `evaluate_lsanomaly.zip`
(see https://cit.mak.ac.ug/staff/jquinn/software/lsanomaly.html).

**usage**: run_eval.py [-h] --data-dir DATA_DIR --output-json JSON_FILE

Perform evaluation of LSAnomaly on downloaded data-sets. 5-fold cross
validation is performed.

**Arguments**

-h, --help
    show this help message and exit

--data-dir DATA_DIR, -d DATA_DIR
    directory of stored data-sets in `libsvm` format

--params YML_PARAMS, -p YML_PARAMS
    YAML file with evaluation parameters

--output-json JSON_FILE, -o JSON_FILE
    path and file name of the results

"""
# The MIT License
#
# Copyright 2019 John Quinn, Chris Skiscim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
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
import json
import logging
import math
import os
import time

import numpy as np
import yaml
from sklearn import (
    model_selection,
    cluster,
    metrics,
    svm,
    neighbors,
    preprocessing,
)
from sklearn.datasets import load_svmlight_file

import lsanomaly

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

fmt = "[%(asctime)s %(levelname)-8s] [%(filename)s:%(lineno)4s - %(funcName)s()] %(message)s"  # noqa
logging.basicConfig(level=logging.DEBUG, format=fmt)


def evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    outlier_class,
    method_name,
    current_method_aucs,
    sigma,
    rho=0.1,
    nu=0.5,
):
    """
    Evaluation for a method and data set. Calculates the AUC for a single
    evaluation fold.

    Args:
        X_train (numpy.ndarray): independent training variables

        y_train (numpy.ndarray): training labels

        X_test (numpy.ndarray): independent test variables

        y_test (numpy.ndarray): test labels

        outlier_class (int): index of the outlier class

        method_name (str): method being run

        current_method_aucs (list): input to the *results* dictionary

        sigma (float): kernel lengthscale for LSAD and OCSVM

        rho (float): smoothness parameter for LSAD

        nu (float): OCSVM parameter - see *scikit-learn* documentation

    Raises:
        ValueError: if a `NaN` is encountered in the AUC calculation.

    """
    try:
        if method_name == "LSAD":
            lsanomaly_model = lsanomaly.LSAnomaly(
                n_kernels_max=500, gamma=sigma ** -2, rho=rho
            )
            lsanomaly_model.fit(X_train, y_train)
            predictions = lsanomaly_model.predict_proba(X_test)[:, -1]

        elif method_name == "OCSVM":
            svm_anomaly_model = svm.OneClassSVM(gamma=sigma ** -2, nu=nu)
            svm_anomaly_model.fit(X_train)
            predictions = 1 - svm_anomaly_model.decision_function(X_test)

        elif method_name == "KNN":
            anomaly_model = neighbors.NearestNeighbors(10)
            anomaly_model.fit(X_train)
            dists, idx = anomaly_model.kneighbors(X_test)
            predictions = dists[:, -1]

        elif method_name == "KM":
            km = cluster.KMeans(min(X_train.shape[0], 20))
            km.fit(X_train)
            nn = neighbors.NearestNeighbors(1)
            nn.fit(km.cluster_centers_)
            dists, idx = nn.kneighbors(X_test)
            predictions = dists[:, 0]

        else:
            raise ValueError("unknown method: {}".format(method_name))

        fpr, tpr, thresholds = metrics.roc_curve(
            y_test == outlier_class, predictions
        )

        metric_auc = metrics.auc(fpr, tpr)
        logger.debug("\tAUC: {:>6.4f}".format(metric_auc))

        if not math.isnan(metric_auc):
            current_method_aucs.append(metric_auc)
        else:
            raise ValueError("NaN encountered in {}".format(method_name))
    except (IndexError, ValueError, Exception) as e:
        logger.exception(
            "\t{} {}: {}".format(method_name, type(e), str(e)), exc_info=True
        )
        raise


def gen_data(data_sets):
    """
    Generator to deliver independent, dependent variables and the name
    of the data set.

    Args:
        data_sets (list): data sets read from the data directory

    Returns:
        numpy.ndarray, numpy.ndarray, str: `X`, `y`, `name`
    """
    for dataset in data_sets:
        path, name = os.path.split(dataset)

        try:
            X, y = load_svmlight_file(dataset)
        except (ValueError, FileNotFoundError, Exception) as e:
            logger.error("unable to load {}".format(dataset))
            logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)
            raise

        X = np.array(X.todense())
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

        classes_ = list(set(y))
        first_two_classes = np.logical_or(y == classes_[0], y == classes_[1])
        X = X[first_two_classes, :]
        y = y[first_two_classes]
        yield X, y, name


def gen_dataset(data_dir):
    """
    Generator for the test data file paths. All
    test files must be capable of being loaded by `load_svmlight_file()`. Files
    with extensions `.bz2`, `.csv` are ignored. Any file beginning with `.`
    is also ignored.

    This walks the directory tree starting at `data_dir`, therefore all
    subdirectories will be read.

    Args:
        data_dir (str): Fully qualified path to the data directory

    Returns:
        list: `svmlight` formatted data sets in `data_dir`

    """
    if not os.path.isdir(data_dir):
        raise ValueError("not a directory: {}".format(data_dir))

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".bz2") or filename.endswith(".csv"):
                continue
            if filename.startswith("."):
                continue
            ds = os.path.join(data_dir, filename)
            if os.path.isdir(ds):
                continue
            yield ds


def _read_params(param_file):
    try:
        with open(param_file) as yml_file:
            params = yaml.safe_load(yml_file)
    except (FileNotFoundError, ValueError):
        raise
    return params


def main(data_dir, json_out, param_file, n_splits=5, rho=0.1, nu=0.5):
    """
    The main show. Loop through all the data-sets and methods running
    a 5-fold stratified cross validation. The results are saved to the
    specified `json_out` file for further processing.

    Args:
        data_dir (str): directory holding the downloaded data sets

        json_out (str): path and filename to store the evaluation results

        param_file (str): YAML file with evaluation parameters

        n_splits (int): number of folds in the cross-validation.

    """
    params = _read_params(param_file)
    method_names = params["evaluation"]["methods"]
    n_methods = len(method_names)

    results = dict()
    results["auc"] = dict()
    results["time"] = dict()
    results["methods"] = method_names[:n_methods]
    results["datasize"] = dict()
    results["n_classes"] = dict()

    # chain the generators
    datasets = gen_dataset(data_dir)
    data_gen = gen_data(datasets)

    logger.debug("starting...")
    ot_start = time.time()

    for X, y, dataset_name in data_gen:
        results["auc"][dataset_name] = [None] * n_methods
        results["time"][dataset_name] = [None] * n_methods

        classes = list(set(y))
        outlier_class = classes[-1]
        y[y != outlier_class] = classes[0]

        classes = list(set(y))
        results["n_classes"][dataset_name] = len(classes)

        results["datasize"][dataset_name] = X.shape
        sigma = lsanomaly.lengthscale_approx.median_kneighbour_distance(X)

        for method, method_name in enumerate(method_names):
            logger.debug(
                "dataset: {}, method: {}".format(dataset_name, method_name)
            )
            current_method_aucs = list()
            start_time = time.time()

            kf = model_selection.StratifiedKFold(n_splits=n_splits)
            for train, test in kf.split(X, y):
                X_train = X[train, :]
                y_train = y[train]
                X_train = X_train[y_train != outlier_class]
                y_train = y_train[y_train != outlier_class]

                X_test = X[test, :]
                y_test = y[test]

                evaluate(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    outlier_class,
                    method_name,
                    current_method_aucs,
                    sigma,
                    rho=rho,
                    nu=nu,
                )
            elapsed = time.time() - start_time
            logger.debug(
                "\tavg AUC    : {:>8.4f}".format(np.mean(current_method_aucs))
            )
            logger.debug("\ttotal time : {:>8.4f}s".format(elapsed))
            logger.debug("\tavg time   : {:>8.4f}s".format(elapsed / n_splits))
            results["time"][dataset_name][method] = time.time() - start_time
            results["auc"][dataset_name][method] = copy.copy(
                current_method_aucs
            )
            with open(json_out, "w") as fp:
                json.dump(results, fp)
    logger.debug(
        "Total evaluation time was about {:>4.2f}m".format(
            (time.time() - ot_start) / 60.0
        )
    )
    logger.debug("Results written to {}".format(json_out))


if __name__ == "__main__":
    """
    Accept command line arguments and kick off the evaluation.
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform evaluation of LSAnomaly on downloaded"
        "data-sets. 5-fold cross validation is performed."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        dest="data_dir",
        required=True,
        help="directory of stored data-sets in libsvm format",
    )
    parser.add_argument(
        "--params",
        "-p",
        dest="yml_params",
        required=True,
        help="YAML file with evaluation parameters",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        dest="json_file",
        required=True,
        help="path and file name of the results (JSON)",
    )
    args = parser.parse_args()

    try:
        sys.exit(
            main(args.data_dir, args.json_file, args.yml_params, n_splits=5)
        )
    except SystemExit:
        pass
