---
description: "Run project tests with pytest, optionally filtered by pattern or file"
agent: "agent"
argument-hint: "Optional: test file path or -k pattern to filter"
---
Run tests for the HAR Edge Deployment Framework.

1. Activate the virtual environment if needed: `.\.venv\Scripts\Activate.ps1`
2. Run pytest:
   - All tests: `python -m pytest tests/ -v`
   - Specific file: `python -m pytest tests/test_main.py -v`
   - Pattern match: `python -m pytest tests/ -v -k "<pattern>"`
3. For code generator changes, also run: `python validate_deployment.py`
4. Report which tests passed/failed with error context.
