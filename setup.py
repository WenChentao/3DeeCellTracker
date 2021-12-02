import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="3DeeCellTracker",
    version="0.4.1",
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
    python_requires='>=3.7',
)
