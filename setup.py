"""
Setup script pour Weli
"""
from setuptools import setup, find_packages
import os

# Lire le README
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Lire les requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='weli',
    version='0.1.0',
    description='Framework de Deep Learning en Python',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Pullo Ba',
    author_email='pulloba192@gmail.com',
    url='',
    packages=find_packages(exclude=['tests', 'testes', 'examples', 'envweli']),
    install_requires=read_requirements(),
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='deep learning, neural networks, machine learning, ai',
    project_urls={
        'Documentation': '',
        'Source': '',
        'Tracker': '',
    },
)
