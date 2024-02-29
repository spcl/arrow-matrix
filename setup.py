from setuptools import setup, find_packages

setup(
    name='arrow',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'scipy>=1.12.0',
        'igraph>=0.11.4',
        'mat73>=0.62',
        'tqdm>=4.66.2',
        'setuptools>=68.2.0',
        'mpi4py>=3.1.5',
        'matplotlib>=3.8.3'
    ],
    entry_points={
        'console_scripts': [
            'arrow_decompose=scripts.decomposition_main:main',
            'spmm_arrow=scripts.spmm_arrow_main:main',
            'spmm_15d=scripts.spmm_15d_main:main',
            'spmm_petsc=scripts.spmm_petsc_main:main',
        ],
    },
)