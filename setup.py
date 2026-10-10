"""
Setup script pour Weli
"""
from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent


def read_requirements():
    return [
        line.strip()
        for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]

setup(
    name='weli-ml',
    version='1.0.2',
    description='Framework de Deep Learning en Python',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Pullo Ba',
    author_email='pulloba192@gmail.com',
    url='https://github.com/Pullo032/Weli',
    packages=find_packages(include=['weli', 'weli.*']),
    install_requires=read_requirements(),
    license='MIT',
    extras_require={
        'test': ['pytest>=7.0'],
    },
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
        'Source': 'https://github.com/Pullo032/Weli',
        'Tracker': 'https://github.com/Pullo032/Weli/issues',
    },
)
