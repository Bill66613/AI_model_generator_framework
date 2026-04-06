# Test Runner

Run tests for the HAR Edge Deployment Framework with appropriate scope.

## Usage
- No arguments: run all tests
- With file path: run specific test file
- With `-k pattern`: run tests matching pattern

## Steps

1. Activate the virtual environment (if not already):
```bash
.\.venv\Scripts\Activate.ps1
```

2. Run tests:
```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_main.py -v

# Pattern match
python -m pytest tests/ -v -k "$PATTERN"
```

3. If tests fail, read the failure output carefully and report:
   - Which tests failed
   - The error type and message
   - Relevant code context

## Parity Testing
For code generator changes, also validate deployment parity:
```bash
python validate_deployment.py
```

$ARGUMENTS
