import pytest
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"sys.path: {sys.path}")

try:
    import hpfracc
    print(f"hpfracc location: {hpfracc.__file__}")
except Exception as e:
    print(f"Failed to import hpfracc: {e}")

# Try to collect tests manually
test_file = "tests/test_ml/test_tensor_ops_comprehensive.py"
print(f"Checking if test file exists: {os.path.exists(test_file)}")

# Run pytest through API
ret = pytest.main(["-v", test_file])
print(f"Pytest return code: {ret}")
sys.exit(ret)
