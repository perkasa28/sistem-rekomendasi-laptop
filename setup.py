from setuptools import setup, find_packages

setup(
    name="sistem-rekomendasi-laptop",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.2",
        "streamlit>=1.12.0",
        "pytest>=7.0.0",
        "python-dotenv>=0.19.0",
        "typing-extensions>=4.0.0"
    ],
)
