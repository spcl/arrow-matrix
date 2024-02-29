#!/bin/bash

# Navigate to the root directory of the project
cd "$(dirname "$0")/.."

export PYTHONPATH=$PYTHONPATH:"./"

python -m unittest tests/test_arrowdecomposition.py
mpiexec -n 30 --oversubscribe python ./tests/test_arrowmpi.py
mpiexec -n 4 --oversubscribe python ./tests/test_spmmPETSc.py
mpiexec -n 6 --oversubscribe python ./tests/test_spmmPETSc.py