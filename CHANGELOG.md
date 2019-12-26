# CHANGE LOG
## Version 1.4
- Move the algorithm out of `__init__()` to a `_lsanomaly.py` in keeping with the
`scikit-learn` structure. This gives proper visibility to the class.
- Move length scale heuristics to `lengthscale_approx.py` to simplify the 
`LSAnomaly` class.
- `LSAnomaly.fit` method now returns `self`.
- Use standard logging throughout.
- Raise a `ValueError` if `X.dim < 2`.
- Raise a `ValueError` in `score` if `y` is `None`.
- Refactor `LSAnomaly.score`.
- In `predict`, change `all_classes.append("anomaly")` to `all_classes.append(1.0)` to
match the original documentation.
- Change the class docstring example from 4 to 5 elements since the code fails in
`median_kneighbor_distance` as the default number of number of neighbors is 5.
- Add `pytest` tests for the examples of v1.2 and README examples. All test data
is serialized as JSON in `tests/data`.
- Include an `evaluate` directory to download test files, run the 5-fold validation,
and create a LaTeX document with a table of results. The scripts can be run from
the command line or programmatically - each of the three scripts has a documented
`main()`.
` Add Jupyter notebooks for sample applications.
- Add a `Makefile` for basic clean up.
- Add an optional `seed` argument to `LSAnomaly` for reproducibility.
- Refactor `setup.py` to allow tests to be run directly from `setup`.  
- Use Google style docstrings.
- Add Sphinx documentation.
- Adhere to PEP8.
- Bump version to 1.4.0

## Test Coverage
```
$ py.test --cov

-- Docs: https://docs.pytest.org/en/latest/warnings.html

---------- coverage: platform darwin, python 3.7.3-final-0 -----------
Name                                  Stmts   Miss  Cover
---------------------------------------------------------
lsanomaly/_lsanomaly.py                 132      4    97%
lsanomaly/lengthscale_approx.py          31      0   100%
lsanomaly/tests/conftest.py              75      0   100%
lsanomaly/tests/test_ecg.py              25      0   100%
lsanomaly/tests/test_example.py          28      0   100%
lsanomaly/tests/test_exceptions.py       17      0   100%
lsanomaly/tests/test_lengthscale.py      13      0   100%
lsanomaly/tests/test_multi_class.py      28      0   100%
lsanomaly/version.py                      1      0   100%
---------------------------------------------------------
TOTAL                                   350      4    99%

============================================== 14 passed, 1 warnings in 1.78s
```