# Building your own Python Library

## Create an virtual environment

python3 -m venv venv
source venv/bin/activate

## install all packages

pip install wheel
pip install setuptools
pip install twine
pip install pytest==4.4.1
pip install pytest-runner==4.4

## Set up folder structure for your package

ResponsibleML
    --aigovernance
        --__init__.py
        --responsibleML.py
    --tests
    --setup.py
    --readme.md

## setup.py code

from setuptools import find_packages, setup
setup(
    name='mypythonlib',
    packages=find_packages(include=['mypythonlib']),
    version='0.1.0',
    description='My first Python library',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)

## Build your library

python setup.py bdist_wheel
pip install /path/to/wheelfile.whl

## upload to pypi

1. Create a login in <https://test.pypi.org/>
2. Upload to testpypi - twine upload -r testpypi dist/*
3. install from testpypi - pip install -i https://test.pypi.org/simple/ responsibleML