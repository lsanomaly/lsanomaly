|Logo|
------
|Language| |License| |Documentation|

lsanomaly is a flexible, fast, probabilistic method for calculating outlier scores on test data, given training examples of inliers. Out of the box it works well with sci-kit learn packages. See the features section for why you might chose this model over other options.

Table of Contents
-----------------

-  `Features`_
-  `Installation`_
-  `Usage`_
-  `Reference`_
-  `Documentation`_
-  `History`_
-  `License`_

Features
--------

-  Compatible with scikit-learn package modules
-  Probabilistic outlier detection model
-  Robust classifier when given multiple inlier classes
-  Easy to install and get started

Installation
------------

The best way to install lsanomaly is to:

::

    pip install lsanomaly

Because lsanomaly requires scikit-learn it also requires numpy and scipy
inherintly. Make sure you have successfully installed these packages if you're
having trouble getting lsanomaly to install.

Usage
-----

For those familiar with scikit-learn the interface will be familiar, in fact lsanomaly was built to be compatible with sklearn modules where applicable. Here is basic usage of lsanomaly to get started quick as possible.

**Configuring the Model**

The LSAnomaly provides reasonable default parameters when given an empty init or it can be passed values for rho and sigma. The value rho controls sensitivity to outliers and sigma determines the ‘smoothness’ of the
boundary. These values can be tuned to improve your results using lsanomaly.

.. code:: python

    from lsanomaly import LSAnomaly

    # At train time lsanomaly calculates parameters rho and sigma
    lsanomaly = LSAnomaly()
    # or alternatively
    lsanomaly = LSAnomaly(sigma=3, rho=0.1)

**Training the Model**

After the model is configured the training data can be fit.

.. code:: python

    import numpy as np
    lsanomaly.fit(np.array([[1],[2],[3],[1],[2],[3]]))

**Making Predictions**

Now that the data is fit, we will probably want to try and predict on some data not in the training set.

.. code:: python

    >>> lsanomaly.predict([0])
    [0.0]
    >>> lsanomaly.predict_proba([0])
    array([[ 0.5760205,  0.4239795]])

Reference
---------

J.A. Quinn, M. Sugiyama. A least-squares approach to anomaly detection in static and sequential data. Pattern Recognition Letters 40:36-40, 2014.  

[`pdf`_]

Documentation
-------------

Coming soon...

History
-------

To check out the complete release notes see the `changelog`_.

License
-------

The MIT License (MIT)

Copyright (c) 2016 John Quinn

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL TH

.. _Features: #features
.. _Installation: #installation
.. _Usage: #usage
.. _Documentation: #documentation
.. _History: #history
.. _License: #license
.. _here: https://
.. _changelog: https://
.. _pdf: http://air.ug/~jquinn/papers/PRLetters_LSAnomalyDetection.pdf

.. |Logo| image:: https://github.com/lsanomaly/lsanomaly/blob/master/docs/logo.png
.. |Language| image:: https://img.shields.io/badge/language-python-blue.svg
.. |Documentation| image:: https://img.shields.io/badge/docs-100%25-brightgreen.svg
.. |License| image:: https://img.shields.io/badge/license-MIT-7f7f7f.svg
