#!/bin/bash
# Post-edit reminder hook
# After editing code files, remind about testing and documentation

FILE="$1"

# Reminder for deployment files
if [[ "$FILE" == *"deployment/"* ]]; then
  echo "📋 Post-edit reminders for deployment code:"
  echo "   1. Check MicroPython generator sync if base_generator changed"
  echo "   2. Run: python -m pytest tests/ -v"
  echo "   3. Consider: python validate_deployment.py"
  echo "   4. Update TECHNICAL_FINDINGS.md if this is a new finding"
fi

# Reminder for callback files
if [[ "$FILE" == *"callbacks/"* ]]; then
  echo "📋 Post-edit reminder: Test with 'python app.py' and verify at http://127.0.0.1:8050"
fi

# Reminder for thesis files
if [[ "$FILE" == *"academic-paper"* ]]; then
  echo "📋 Post-edit reminder: Compile with 'cd academic-paper-vietnamese && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex'"
fi

exit 0
