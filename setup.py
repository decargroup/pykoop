import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name='pykoop',
    version='0.1.0',
    description='Koopman operator identification library in Python',
    long_description=readme,
    author='Steven Dahdah, Vassili Korotkine',
    author_email=('steven.dahdah@mail.mcgill.ca, '
                  'vassili.korotkine@mail.mcgill.ca'),
    url='https://bitbucket.org/decargroup/pykoop',
    packages=setuptools.find_packages(exclude=('tests', 'examples')),
    install_requires=[
        'numpy', 'scipy', 'scikit-learn', 'picos', 'pandas', 'optht'
    ],
    extra_require={'MOSEK solver': ['mosek']},
)
