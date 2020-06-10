import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metcalpy",
    version="0.0.1",
    author="METplus",
    author_email="met-help@ucar.edu",
    description="statistics and util package for METplus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NCAR/METcalcpy",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)