from setuptools import setup, find_packages

setup(
  name = 'nystrom-attention',
  packages = find_packages(),
  version = '0.0.9',
  license='MIT',
  description = 'Nystrom Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/nystrom-attention',
  keywords = [
    'artificial intelligence',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)