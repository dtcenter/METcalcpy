import setuptools
from setuptools import setup, find_packages
from distutils.util import convert_path

with open("README.md", "r") as fh:
    long_description = fh.read()

main_ns = {}
version_path = convert_path('docs/version')
with open(version_path) as version_file:
    exec(version_file.read(), main_ns)

setuptools.setup(
    name="metcalcpy",
    version=main_ns['__version__'],
    author="METplus",
    description="statistics and util package for METplus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dtcenter/METcalcpy",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
