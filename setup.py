from setuptools import setup, find_packages

setup(
  name="localvectorstore",
  version="0.1.0",
  packages=find_packages(),
  install_requires=[
      "numpy",
      "langchain",
  ],
  author="Sai Yashwanth",
  author_email="taddishetty34@gmail.com",
  description="A simple local vector store implementation using numpy",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/theyashwanthsai/localvectorstore",
)
