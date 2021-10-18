import setuptools

with open('README.rst', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='pykoop',
    version='1.0.0',
    description='Koopman operator identification library in Python',
    long_description=readme,
    author='Steven Dahdah',
    author_email='steven.dahdah@mail.mcgill.ca',
    url='https://github.com/decarsg/pykoop',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    packages=setuptools.find_packages(exclude=('tests', 'examples', 'doc')),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.2.1',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.1',
        'picos>=2.2.52',
        'pandas>=1.3.1',
        'optht>=0.2.0',
    ],
    extra_require={
        'MOSEK solver': ['mosek>=9.2.49'],
        'SMCP solver': ['smcp>=0.4.6'],
    },
)
