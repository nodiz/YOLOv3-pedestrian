from setuptools import setup, find_packages

# TODO: ADD ALL dependencies from your module and version them
# You can then install the package with pip install -e PATH TO MODULE

setup(
    name="detlib",
    version="DEV",
    description="Retriever Module",
    author="...",
    author_email='...',
    packages=find_packages(include=["detlib", "detlib.*"]),
    install_requires=[
        "fire",     # could be useful for your CLI if you want to move train and test scripts inside the package and use CLI args
        "flask",
        "torch>=1.4",
        "torchvision",
        "numpy",
        "matplotlib",
        "tensorflow",
        "tensorboard",
        "terminaltables",
        "pillow",
        "tqdm",
        "ADD-ALL-YOUR-PACKAGE-DEPS (./detlib folder deps)",
    ],
    python_requires=">=3.7,<4.0",
)
