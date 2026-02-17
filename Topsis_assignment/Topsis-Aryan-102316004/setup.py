from setuptools import setup, find_packages

setup(
    name="Topsis-Aryan-102316004",
    version="1.0.0",
    author="Aryan Thakur",
    author_email="aryan@example.com",
    description="A Python package for TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aryan/topsis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "topsis=topsis.main:main",
        ],
    },
)
