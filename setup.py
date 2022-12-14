"""``pykoop`` setup file."""

import setuptools

with open('README.rst', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='pykoop',
    version='1.1.0',
    description='Koopman operator identification library in Python',
    long_description=readme,
    author='Steven Dahdah',
    author_email='steven.dahdah@mail.mcgill.ca',
    url='https://github.com/decargroup/pykoop',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    project_urls={
        'Documentation': 'https://pykoop.readthedocs.io/en/latest',
        'Source': 'https://github.com/decargroup/pykoop',
        'Tracker': 'https://github.com/decargroup/pykoop/issues',
        'PyPI': 'https://pypi.org/project/pykoop/',
        'DOI': 'https://doi.org/10.5281/zenodo.5576490',
    },
    packages=setuptools.find_packages(exclude=('tests', 'examples', 'doc')),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'picos>=2.4.0',
        'pandas>=1.3.1',
        'optht>=0.2.0',
        'Deprecated>=1.2.13',
        'matplotlib>=3.5.1',
    ],
    extra_require={
        'MOSEK solver': ['mosek>=9.2.49'],
        'CVXOPT solver': ['cvxopt>=1.3.0'],
        'SMCP solver': ['smcp>=0.4.6'],
    },
)
