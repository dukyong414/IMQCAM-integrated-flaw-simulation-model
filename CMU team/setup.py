from setuptools import setup, find_packages
import os

# Utility to read the README file for the long description
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

# Read requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line and not line.startswith('#')]

setup(
    name='porosity_modeling',
    version='0.1.0',
    author='Justin Miner',
    author_email='jpminer@andrew.cmu.edu',
    description='Modular porosity modeling framework for laser powder bed fusion',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/CMU-EMIT-Lab/PorosityModel',
    packages=find_packages(where='.'),  # Adjust if your code is under a subdirectory
    python_requires='>=3.8',
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'run_porosity_model=run_model:main',
            'porosity_gui=main_window:main',
        ],
    },
)
