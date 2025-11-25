#!/usr/bin/env python3
"""
Comprehensive Example Testing Script
Tests all examples in the examples/ folder and reports status
"""

import subprocess
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Set environment variables for non-interactive testing
os.environ['MPLBACKEND'] = 'Agg'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

def run_example_test(example_path, timeout=120):
    """Test a single example file."""
    try:
        # Get project root (parent of tests directory)
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ['python', str(example_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root)
        )
        
        return {
            'status': 'PASS' if result.returncode == 0 else 'FAIL',
            'exit_code': result.returncode,
            'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'status': 'TIMEOUT',
            'exit_code': -1,
            'stdout': '',
            'stderr': f'Test timed out after {timeout} seconds'
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'exit_code': -1,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    # Get project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    base_path = project_root / 'examples'
    
    # List of all Python examples to test
    example_files = list(base_path.rglob('*.py'))
    
    # Filter out __pycache__ and other non-example files
    example_files = [f for f in example_files if '__pycache__' not in str(f)]
    
    # Sort for consistent ordering
    example_files.sort()
    
    print(f"{'='*80}")
    print(f"TESTING {len(example_files)} EXAMPLE FILES")
    print(f"{'='*80}\n")
    
    results = {}
    passed = 0
    failed = 0
    timeout = 0
    error = 0
    
    for i, example in enumerate(example_files, 1):
        rel_path = example.relative_to(base_path)
        print(f"[{i}/{len(example_files)}] Testing {rel_path}...", end=' ', flush=True)
        
        result = run_example_test(example)
        results[str(rel_path)] = result
        
        status = result['status']
        if status == 'PASS':
            print(f"✅ PASS")
            passed += 1
        elif status == 'FAIL':
            print(f"❌ FAIL (exit code: {result['exit_code']})")
            failed += 1
        elif status == 'TIMEOUT':
            print(f"⏱️  TIMEOUT")
            timeout += 1
        else:
            print(f"⚠️  ERROR")
            error += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total examples tested: {len(example_files)}")
    print(f"✅ Passed: {passed} ({passed/len(example_files)*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/len(example_files)*100:.1f}%)")
    print(f"⏱️  Timeout: {timeout} ({timeout/len(example_files)*100:.1f}%)")
    print(f"⚠️  Error: {error} ({error/len(example_files)*100:.1f}%)")
    
    # Print failed examples with error messages
    if failed > 0 or timeout > 0 or error > 0:
        print(f"\n{'='*80}")
        print("FAILED/TIMEOUT/ERROR EXAMPLES:")
        print(f"{'='*80}")
        for rel_path, result in results.items():
            if result['status'] != 'PASS':
                print(f"\n{rel_path}: {result['status']}")
                if result['stderr']:
                    error_lines = result['stderr'].split('\n')
                    # Print last 10 lines of error
                    print("  Error (last 10 lines):")
                    for line in error_lines[-10:]:
                        print(f"    {line}")
    
    # Save detailed results to JSON
    project_root = Path(__file__).parent.parent
    output_file = project_root / 'example_test_results.json'
    with open(str(output_file), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': len(example_files),
                'passed': passed,
                'failed': failed,
                'timeout': timeout,
                'error': error
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file.absolute()}")
    
    return 0 if (failed + timeout + error) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())

