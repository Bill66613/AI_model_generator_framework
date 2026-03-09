---
description: "Resume work from a previous session by reading project status, recent git history, and TODO items"
agent: "agent"
---
Resume work by loading context from the project:

1. Check git status and recent history:
   ```bash
   git status --short
   git branch --show-current
   git log --oneline -5
   git worktree list
   ```
2. Read current status files:
   - `TODO.md`
   - `academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md` (thesis progress)
   - `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` (Quick Context section only)
3. Report:
   - Current branch and uncommitted changes
   - Active worktrees
   - What was last worked on (from git log)
   - Top priority items from TODO
   - Suggested next steps
