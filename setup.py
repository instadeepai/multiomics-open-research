import os

from setuptools import find_packages, setup

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="multiomics-open-research",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/instadeepai/multiomics-open-research",
    license="Apache-2.0",
    author="InstaDeep Ltd and CentraleSupélec",
    python_requires=">=3.10",
    description="BulkRNABert, MOJO and CellScribe: language models for bulk RNA-seq, "
    "DNA methylation and single-cell RNA-seq",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "click==8.1.7",
        "jax==0.4.19",
        "jaxlib==0.4.19",
        "dm-haiku==0.0.10",
        "joblib==1.3.2",
        "numpy==1.26",
        "pandas==2.2.0",
        "python-dateutil<3.0.0",
        "pydantic==1.10.13",
        "jupyter",
        "huggingface-hub>=0.30.0",
        "matplotlib==3.10.9",
        # CellScribe (PyTorch). transformers 5.3.0 does not tie CellScribe's
        # output embeddings.
        "torch>=2.6",
        "transformers>=4.51,!=5.3.0",
        "safetensors",
    ],
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_releases.html",
    ],
    keywords=[
        "Multi-omics",
        "Language Model",
        "Deep Learning",
        "JAX",
        "PyTorch",
        "Single-cell",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
