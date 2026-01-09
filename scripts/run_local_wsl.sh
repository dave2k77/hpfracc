#!/bin/bash
source .venv_wsl/bin/activate
PROJECT_ROOT=$(pwd)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd tests/test_ml
python3 -m pytest test_tensor_ops_comprehensive.py -v
