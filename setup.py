from setuptools import find_packages, setup
setup(
    name='neurallogic',
    packages=find_packages(include=['neurallogic']),
    version='0.1.0',
    description='A Neural Logic Library',
    author='@z80coder',
    install_requires=[],
    setup_requires=['pytest-runner'],
    test_suite='tests',
)