import os
from setuptools import setup

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()

setup(
    name="epuv",
    version=0.1,
    license="MIT license",
    description="Calibration Tests for Ensemble Predictors.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Mira Juergens",
    author_email="mira.juergens@ugent.be",
    url="https://github.com/mkjuergens/EnsembleCalibration",
    packages=["ensemblecalibration"],
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "setuptools",
        "matplotlib",
        "ternary",
        "pandas",
        "seaborn",
        "statsmodels",
        "tqdm",
        "composition_stats",
        "torch"

    ],
    include_package_data=True,
)
