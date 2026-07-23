from setuptools import setup, find_packages

setup(
    name="uq_toolbox",
    version="0.1.0",
    description="Uncertainty Quantification Toolbox for Text Mining",
    packages=find_packages(),
    install_requires=[
        # Inserisci qui le tue dipendenze se necessarie (es. "transformers", "datasets")
    ],
    python_requires=">=3.8",
)
