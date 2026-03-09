---
name: session-continuity
description: 'Save and restore session context for cross-session continuity. Use when handing off work between sessions, resuming from a previous session, or tracking what was accomplished. Includes session notes, TODO tracking, and technical findings documentation.'
argument-hint: 'Action: handoff (save) or resume (load)'
---

# Session Continuity

## When to Use
- End of a work session — save context for next time
- Start of a new session — load what happened last time
- Switching between worktrees/branches — track what each branch is doing

## Handoff Procedure (End of Session)

### 1. Summarize accomplishments
List what was done, files changed, decisions made.

### 2. Update tracking files
- `TODO.md` — Mark completed items, add new ones
- `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` — If any new technical discovery
- `academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md` — If thesis progress

### 3. Git commit with descriptive message
```bash
git add -A
git commit -m "session: {brief summary of what was done}"
```

### 4. Note blockers and next steps
Record anything the next session needs to know.

## Resume Procedure (Start of Session)

### 1. Check environment
```bash
git branch --show-current
git status --short
git log --oneline -5
git worktree list
```

### 2. Read status files
- `TODO.md` — What's pending
- `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` — Quick Context section (top)
- `academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md` — Thesis progress

### 3. Check for in-progress work
Look at recent git log for uncommitted changes or WIP commits.

### 4. Report to user
- Current branch and status
- What was last worked on
- Suggested next steps
- Active worktrees (if any)

## Tracking Files

| File | Purpose | Update When |
|------|---------|-------------|
| `TODO.md` | Project tasks | Any session |
| `TECHNICAL_FINDINGS.md` | Parity bugs, novel discoveries | Code changes |
| `THESIS_REPORT_INSTRUCTIONS.md` | Chapter progress, terminology | Thesis work |
