# Session Resume — Pick up from a previous session

Read context from previous sessions and resume work.

## Steps

1. Check for session notes:
```bash
ls .claude/sessions/ 2>/dev/null || echo "No session notes found"
```

2. Read the most recent session note for context

3. Read current status files:
- `TODO.md` — Project-level tasks
- `academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md` — Thesis progress
- `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` — Technical evidence (read Quick Context section)

4. Check git status for uncommitted work:
```bash
git status
git log --oneline -5
git worktree list
```

5. Report to user:
- What was done in the last session
- What's in progress
- Suggested next steps

$ARGUMENTS
