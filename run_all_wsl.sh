#!/bin/bash
source .venv_wsl/bin/activate
export PYTHONPATH=".:$PYTHONPATH"

echo "--- 1. Environment Info ---"
python3 -c "import sys; print('Python Path:', sys.path)"

echo "--- 2. Import Check ---"
python3 -c "import torch; import jax; import hpfracc; print('hpfracc and backends imported successfully')"

echo "--- 3. File Check ---"
ls -l tests/test_ml/test_tensor_ops_comprehensive.py

echo "--- 4. Pytest Collection Debug ---"
python3 -m pytest tests/test_ml/test_tensor_ops_comprehensive.py --collect-only -v

echo "--- 5. Running Tests ---"
python3 -m pytest tests/test_ml/test_tensor_ops_comprehensive.py -v
