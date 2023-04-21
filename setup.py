import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="3DeeCellTracker",
    version="0.5.0-alpha",
    author="Chentao Wen",
    author_email="chintou.on@gmail.com",
    description="A package for tracking cells in 3D time lapse images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenChentao/3DeeCellTracker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
	"Environment :: GPU :: NVIDIA CUDA"
    ],
    install_requires=[
        "numpy==1.22.3",
        "scikit-learn==1.1.2",
        "tifffile==2023.2.28",
        "tqdm==4.64.0",
        "matplotlib==3.5.1",
        "scipy==1.8.0",
        "pillow==9.1.0",
        "scikit-image==0.15.0",
        "stardist==0.8.3",
        "notebook==6.4.10"
    ],
    python_requires='>=3.8',
)
