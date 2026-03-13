#!/usr/bin/env python3
"""
Script to fix import statements in generated flatbuffer Python files.
Changes absolute imports like 'from FXGraph.KeyValue import KeyValue' 
to relative imports like 'from .KeyValue import KeyValue'.
"""

import glob
import os
import re


def fix_imports_in_file(file_path):
    """Fix import statements in a single Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match imports like 'from FXGraph.Something import Something'
    # and replace with 'from .Something import Something'
    pattern = r'from\s+FXGraph\.(\w+)\s+import\s+(\w+)'
    replacement = r'from .\1 import \2'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed imports in {file_path}")
    else:
        print(f"No imports to fix in {file_path}")


def main():
    """Fix imports in all generated Python files in the FXGraph directory."""
    fxgraph_dir = "python/teenygrad/graph/FXGraph"
    
    if not os.path.exists(fxgraph_dir):
        print(f"Directory {fxgraph_dir} does not exist")
        return
    
    # Find all Python files in the FXGraph directory
    python_files = glob.glob(os.path.join(fxgraph_dir, "*.py"))
    
    for py_file in python_files:
        if py_file.endswith("__init__.py"):
            continue  # Skip __init__.py files
        fix_imports_in_file(py_file)


if __name__ == "__main__":
    main()
