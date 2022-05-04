import setuptools
from setuptools import find_packages, setup
    
setuptools.setup(
    name='responsibleML',
    packages=setuptools.find_packages(),
    version='1.0.0',
    description='responsibleML',
    author='Meenakshisundaram.t@gmail.com',
    license='MIT',
    install_requires=['pytest-runner', 'pandas', 'numpy', 'codecarbon', 'opacus', 'captum'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    python_requires='>=3.6',
)