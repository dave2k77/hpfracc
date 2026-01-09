#!/bin/bash
source .venv_wsl/bin/activate
export PYTHONPATH=".:$PYTHONPATH"
python3 -m pytest tests/test_ml/test_tensor_ops_comprehensive.py -v
