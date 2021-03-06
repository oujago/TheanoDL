from setuptools import find_packages
from setuptools import setup

version = '0.1.0'

description = """
Deep Learning Framework on the top of Theano, so it called "thdl".
"""

install_requires = [
    'numpy',
    'matplotlib',
    'xlwt',
    'nltk',
    'theano',
    'scipy'
]

setup(
    name='thdl',
    version=version,
    description=description,
    author='Chao-Ming Wang',
    packages=find_packages(),
    author_email='oujago@gmail.com',
    url='https://oujago.github.io/',
    install_requires=install_requires
)
