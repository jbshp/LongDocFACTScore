from setuptools import setup

setup(
    name="longdocfactscore",
    version="0.1.0",
    description="a framework for evaluating factual consistency of long document abstractive summarisation",
    url="https://github.com/jbshp/LongDocFACTScore",
    install_requires=[
        "nltk",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers"
    ]
)