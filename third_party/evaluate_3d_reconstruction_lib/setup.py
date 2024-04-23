import os

from setuptools import find_packages, setup


def read(fname):
    """Utility function to read the README file.

    Used for the long_description.  It's nice, because now 1) we have a top
    level README file and 2) it's easier to type in the README file than to put
    a raw string in below.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='evaluate_3d_reconstruction',
    packages=find_packages(include=['evaluate_3d_reconstruction']),
    version='0.1.0',
    description='Evaluates meshes using the F-score. \
        Adapted from Tanks and Temples evaluation script.',
    long_description=read('README.md'),
    author='Erik Sandstroem',
    install_requires=['numpy'],
)
