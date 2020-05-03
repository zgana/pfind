# pfind/setup.py

from setuptools import setup
import sys
import setuptools

__version__ = '0.1.0'


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pfind',
    version=__version__,
    author='Mike Richman',
    author_email='mike.d.richman@gmail.com',
    packages = ['pfind'],
    description='basic particle tracking algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zgana/pfind',
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib'],
)
