from setuptools import setup, find_packages
from codecs import open

with open('README.rst', 'r', 'utf-8') as f:
    readme = f.read()

install_requires = ['scikit-learn']

setup(
    name='lsanomaly',
    version='0.1.1',
    url='https://github.com/lsanomaly/lsanomaly',
    description='Least squares anomaly detection.',
    long_description=readme,
    author='John Quinn',
    author_email='jquinn@cit.ac.ug',
    maintainer='David Westerhoff',
    maintainer_email='dmwesterhoff@gmail.com',
    license='MIT',
    keywords='anomaly outlier novelty detection ' +
             'machine learning scikit-learn sklearn',
    packages=find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    entry_points={
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=install_requires,
)
