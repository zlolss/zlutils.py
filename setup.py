import setuptools

discription = "a package of utils"

setuptools.setup(
  name="zlutils",
  version='0.1.1',
  python_requires=">=3.6",
  author="zlols",
  author_email="zlols@foxmail.com",
  description=discription,
  long_description=discription,
  long_description_content_type="text/markdown",
  url="https://github.com/zlolss/zlutils.py.git",
  py_modules=['zlutils'],
  install_requires=[
    ],
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3 :: Only",
  "License :: OSI Approved :: MIT License",
  #"Operating System :: OS Independent",
  ],
)
