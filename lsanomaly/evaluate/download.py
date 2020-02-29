"""
**download.py**

A commandline utility to retrieve test data from
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ for use in evaluating
LSAnamoly.

**usage**: download.py [-h] --params YML_PARAMS --data-dir DATA_DIR
    [--sc-url SC_URL] [--mc-url MC_URL]

Retrieve datasets for LsAnomaly evaluation. By default, data is retrieved from
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

**Arguments**

-h, --help
    show this help message and exit

--params YML_PARAMS, -p YML_PARAMS
    YAML file with evaluation parameters

--data-dir DATA_DIR, -d DATA_DIR
    directory to store retrieved data sets

--sc-url SC_URL
    optional: single class test data URL; default:
    https:/ /www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/

--mc-url MC_URL
    optional: Multi-class test data URL; default:
    https:// www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/

"""
# The MIT License
#
# Copyright 2019 Chris Skiscim
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
import bz2
import logging
import os

import requests
import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

fmt = "[%(asctime)s %(levelname)-8s] [%(filename)s:%(lineno)4s - %(funcName)s()] %(message)s"  # noqa
logging.basicConfig(level=logging.INFO, format=fmt)


def unzip_write(file_path):
    """
    Reads and inflates a .bz2 file and writes it back.
    The compressed file is retrained. Used internally.

    Args:
        file_path (str): file to inflate

    Raises:
        FileNotFoundError
    """
    try:
        with open(file_path[:-4], "wb") as new_file, bz2.BZ2File(
            file_path, "rb"
        ) as file:
            for data in iter(lambda: file.read(100 * 1024), b""):
                new_file.write(data)
    except (FileNotFoundError, Exception) as e:
        logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)
        raise


def write_contents(file_path, get_request):
    """
    Writes the contents of the get request to the specified file path.

    Args:
        file_path (str): file path

        get_request (requests.Response): response object

    Raises:
        IOError
    """
    try:
        open(file_path, "wb").write(get_request.content)
        if file_path.endswith("bz2"):
            unzip_write(file_path)
    except (IOError, Exception) as e:
        logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)
        raise


def get_request(dataset, file_path, sc_url, mc_url):
    """
    Retrieve *dataset* trying first at `sc_url` and failing that, at
    `mc_url`. If a data set cannot be retrieved, it is skipped.
    The contents to `file_path` with the data set name as the file name.

    Args:
        dataset (str):  Dataset name as referenced in
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

        file_path (str): Directory where `dataset` will be written.

        sc_url (str): single class data set URL

        mc_url (str): multiclass data set URL

    """
    url_get = sc_url + dataset
    try:
        get_req = requests.get(url_get, allow_redirects=True)
    except (requests.exceptions.InvalidURL, Exception) as e:
        logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)
        raise

    if get_req.status_code == 200:
        write_contents(file_path, get_req)
    else:
        url_get = mc_url + dataset
        get_req = requests.get(url_get, allow_redirects=True)
        if get_req.status_code == 200:
            write_contents(file_path, get_req)
        else:
            logger.error("\tunable to retrieve {}".format(dataset))
    logger.info("\tsuccess".format(dataset))


def main(param_file, sc_url, mc_url, data_fp):
    """
    The main show. Tries to retrieve and store all the configured data-sets.

    Args:
        param_file (str): `.yml` File containing the evaluation parameters

        sc_url (str):  single class data set URL

        mc_url (str): multiclass data set URL

        data_fp (str): Directory where the datasets will be written

    Raises:
        ValueError: If `data_fp` is not a valid directory.
    """
    try:
        with open(param_file) as yml_file:
            params = yaml.safe_load(yml_file)
    except (FileNotFoundError, ValueError):
        raise

    datasets = params["evaluation"]["datasets"]

    if not os.path.isdir(data_fp):
        raise ValueError("no directory named {}".format(data_fp))
    try:
        for dataset in sorted(datasets):
            logger.info("retrieving {}".format(dataset))
            write_path = os.path.join(data_fp, dataset)
            get_request(dataset, write_path, sc_url, mc_url)
    except Exception as e:
        logger.exception("{}: {}".format(type(e), str(e)), exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    import sys

    _sc_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    _mc_url = (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
    )
    parser = argparse.ArgumentParser(
        description="Retrieve datasets for LsAnomaly evaluation. "
        "By default, data is retrieved from "
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
    )
    parser.add_argument(
        "--params",
        "-p",
        dest="yml_params",
        required=True,
        help="YAML file with evaluation parameters",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        dest="data_dir",
        required=True,
        help="directory to store retrieved data sets",
    )
    parser.add_argument(
        "--sc-url",
        dest="sc_url",
        required=False,
        default=_sc_url,
        help="optional: single class test data URL; default: {}".format(
            _sc_url
        ),
    )
    parser.add_argument(
        "--mc-url",
        dest="mc_url",
        required=False,
        default=_mc_url,
        help="optional: Multi-class test data URL; default: {}".format(_mc_url),  # noqa
    )
    args = parser.parse_args()
    try:
        sys.exit(main(args.yml_params, args.sc_url, args.mc_url, args.data_dir))  # noqa
    except SystemExit:
        pass
