import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires_list = open(f'zzd/requirements.txt', 'r', encoding='utf8').readlines()
requires_list = [i.strip() for i in requires_list]

setuptools.setup(
    name="zzd",
    version="1.0.5",
    author="ChenPing L", 
	author_email="1965770446@qq.com",
    description="some tools", 
	long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miderxi/zzd_lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requires_list,
)

