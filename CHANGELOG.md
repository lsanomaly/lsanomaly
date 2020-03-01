# CHANGELOG
## v0.1.5
#### C Skiscim
- Remove `pair_centile_distance()` from `lengthscale_approx.py` as it was unused.
- Update `requirements.txt` to most current versions of everything.
- Remove dead links; fix typos; less garish colors in `static_mix.py`.
- Make the _x_-axes line up in plotting `dynamic.py`.
- Fix logging format error in the test scripts.
- Change `pythonpackage.yml` to have `python-version: [3.6, 3.7]` (fixes package check failure).
- Various PEP8 corrections for a clean `flake8` run; reformat text where appropriate.
- Bump version to 0.1.5
## v0.1.4
#### C Skiscim
- Move the algorithm out of `__init__` to  `_lsanomaly.py` in keeping with the
`scikit-learn` structure. This gives proper visibility to the code.
- Add convenience `import`s to the package `__init__`, a la scikit-learn.
- Move length scale heuristics to `lengthscale_approx.py` to simplify the 
`LSAnomaly` class.
- `LSAnomaly.fit` method now returns `self`.
- Use standard logging throughout.
- Raise exceptions when it makes sense.
- Refactor `LSAnomaly.score` - possible bug.
- Raise `NotImplementedError` for `get_/set_params` (for now).
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
- Add Jupyter notebooks for sample applications.
- Add a `Makefile` for basic clean up.
- Add an optional `seed` argument (RNG) to `LSAnomaly` for reproducing results.
- Refactor `setup.py` to allow tests to be run directly from `setup`.  
- Use Google style docstrings.
- Add Sphinx documentation.
- Adhere to PEP8.
- Bump version to 1.4.0

## v0.1.3
#### D Westerhoff
- Package for PyPy, etc.

### Test Coverage
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

```
