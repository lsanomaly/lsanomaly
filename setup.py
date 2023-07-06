# flake8: noqa
# pylint: skip-file

import re
import sys
from os import path

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

__version__ = None
exec(open("lsanomaly/version.py").read())


here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    split = re.split("(Table of Contents)", f.read())
    LONG_DESCRIPTION = split[1] + split[2]
    LONG_DESCRIPTION_TYPE = "text/markdown"

DATA = {"lsanomaly/evaluate": ["eval_params.yml"],
        "lsanomaly/notebooks": ["filtered_ecg.json"]}


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = list()
        self.test_suite = True

    def run_tests(self):
        try:
            import pytest
        except ImportError as e:
            print("pytest must be installed to run tests.")
            print("{}: {}".format(type(e), str(e)))
            raise

        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name="lsanomaly",
    python_requires=">=3.6.8",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    package_data=DATA,
    zip_safe=False,
    tests_require=["pytest"],
    install_requires=[
        "matplotlib==3.1.2",
        "numpy==1.18.0",
        "pyaml>=19.4",
        "requests==2.22.0",
        "scikit-learn==0.21.3",
        "scipy==1.10.0",
    ],
    cmdclass={"test": PyTest},
    url="https://github.com/lsanomaly/lsanomaly",
    author_email="John Quinn <jquinn@cit.ac.ug>, David Westerhoff <dmwesterhoff@gmail.com>, Chris Skiscim <christoph@protonmail.com>",
    description="Least squares anomaly detection.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_TYPE,
    keywords="anomaly outlier novelty detection "
    + "machine learning scikit-learn sklearn",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.7",
    ],
    extras_require={"testing": ["pytest"]},
)
