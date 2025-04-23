from setuptools import setup, find_packages

setup(
    name="RingDetectionToolkit",
    version="0.1.0",
    author="Alessandro Fiorentino",
    author_email="alexfiore98@gmail.com",
    description="A toolkit for ring detection tasks.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0,<1.27.0",
        "matplotlib==3.7.1",
        "scikit-learn==1.2.2",
        "scipy==1.10.1",
        "tqdm==4.65.0",
        # Uncomment the following lines if GPU or deep learning dependencies are needed
        # "pycuda==2025.1",
        # "tensorflow==2.18.0",
    ],
    python_requires=">=3.7",
)
