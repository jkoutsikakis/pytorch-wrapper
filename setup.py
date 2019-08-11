import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='pytorch-wrapper',
    version='1.0.0',
    description='PyTorchWrapper is a library that provides a systematic and extensible way to build, train, evaluate, '
                'and tune deep learning models using PyTorch. It also provides several ready to use modules and '
                'functions for fast model development.',
    long_description=README,
    long_description_content_type='text/markdown',
    author='John Koutsikakis',
    author_email="jkoutsikakis@gmail.com",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    url='https://github.com/jkoutsikakis/pytorch-wrapper',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy>=1.14,<2',
        'scikit-learn>=0.19,<1',
        'six>=1.11,<2',
        'torch>=1,<2',
        'tqdm>=4.27.0,<5',
        'hyperopt>=0.1,<0.2'
    ],
    tests_require=['nose'],
    test_suite='nose.collector'
)
