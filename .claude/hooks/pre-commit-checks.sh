#!/bin/bash
# Pre-commit checks
# Run tests and basic validation before allowing commit

echo "🔍 Running pre-commit checks..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "⚠️  Virtual environment not activated. Activating..."
  source venv/Scripts/activate 2>/dev/null || source venv/bin/activate 2>/dev/null
fi

# Run tests
echo "📋 Running pytest..."
python -m pytest tests/ -v --tb=short
TEST_EXIT=$?

if [ $TEST_EXIT -ne 0 ]; then
  echo "❌ Tests failed! Fix before committing."
  exit 1
fi

# Check for common issues
echo "📋 Checking for common issues..."

# Check for hardcoded paths
if git diff --cached --name-only | xargs grep -l "persistent_data" 2>/dev/null | grep -v config/config.py | grep -v CLAUDE.md | grep -v ".md$" > /dev/null 2>&1; then
  echo "⚠️  Warning: Found hardcoded 'persistent_data' paths outside config.py"
  echo "   Use: from config.config import PERSISTENT_DIR"
fi

# Check for zero-padding references in new code
if git diff --cached | grep -i "zero.pad\|zero_pad\|np.zeros.*pad" > /dev/null 2>&1; then
  echo "⚠️  Warning: Found potential zero-padding code. Use edge-value replication instead!"
fi

echo "✅ Pre-commit checks passed."
exit 0
