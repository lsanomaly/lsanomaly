
clean:
	find . -name \*.pyc -delete
	find . -name __pycache__ -delete
	find . -name .coverage -delete
	rm -rf .cache/
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf lsanomaly/notebooks/.ipynb_checkpoints/
