# Status Check — Quick project health overview

Quickly check the health of the project: git status, test results, and pending work.

## Steps

1. Git status:
```bash
git status --short
git branch --show-current
git log --oneline -3
```

2. Check for active worktrees:
```bash
git worktree list
```

3. Run a quick test:
```bash
python -m pytest tests/ -v --tb=line -q
```

4. Check TECHNICAL_FINDINGS.md for any items marked as "⚠️" or "IN PROGRESS"

5. Check TODO.md for pending tasks

6. Report summary:
   - Current branch and its status
   - Active worktrees
   - Test results (pass/fail count)
   - Top priority items from TODO.md

$ARGUMENTS
