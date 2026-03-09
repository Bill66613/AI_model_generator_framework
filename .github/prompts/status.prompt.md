---
description: "Quick project health check: git status, test results, active worktrees, and pending TODO items"
agent: "agent"
---
Perform a quick health check of the HAR Edge Deployment Framework:

1. Show current git branch and recent commits:
   ```bash
   git branch --show-current
   git log --oneline -5
   ```
2. List active worktrees: `git worktree list`
3. Run a quick test: `python -m pytest tests/ -v --tb=line -q`
4. Check `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` for any items marked "⚠️" or "IN PROGRESS"
5. Check `TODO.md` for pending tasks
6. Summarize: branch name, worktree count, test pass/fail, top priority items.
