from setuptools import setup, find_packages

setup(
    name='itu_auction',
    version='0.1',
    description='Auction algorithms for assignment problems with non-separable valuations implemented in PyTorch',
    packages=find_packages(),
    python_requires='>=3.7',
)
