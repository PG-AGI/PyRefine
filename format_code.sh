#!/bin/bash

# Run Black (Code Formatter)
echo "Running black..."
black test_py_scripts/main.py

# Run Isort (Import Sorting)
echo "Running isort..."
isort test_py_scripts/main.py

# Run Autoflake (Remove Unused Imports and Variables)
echo "Running autoflake..."
autoflake --in-place --remove-all-unused-imports test_py_scripts/main.py

# Run Autopep8 (PEP 8 Formatting)
echo "Running autopep8..."
autopep8 --in-place --aggressive test_py_scripts/main.py

# Run Unimport (Remove Unused Imports)
echo "Running unimport..."
unimport test_py_scripts/main.py

echo "All commands executed successfully!"
