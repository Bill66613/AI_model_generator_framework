---
name: worktree-management
description: 'Manage git worktrees for parallel agent sessions. Use when creating worktrees, listing active branches, merging completed work, or cleaning up finished worktrees. Enables multiple Copilot or Claude Code sessions on separate branches without blocking each other.'
argument-hint: 'Action: create, list, merge, or remove'
---

# Git Worktree Management

## When to Use
- Need to work on a feature while another session handles thesis writing
- Want parallel agent sessions that don't block each other
- Merging a completed feature branch back to main

## Concept
Each worktree = separate working directory + its own branch, sharing the same `.git`. Multiple VS Code windows can each have their own Copilot agent session.

```
D:\...\GUI_app\                      ← main (Session 1)
D:\...\GUI_app-feature-cnn-fix\      ← feature/cnn-fix (Session 2)
D:\...\GUI_app-thesis-chapter-3\     ← thesis/chapter-3 (Session 3)
```

## Procedures

### Create a Worktree
```powershell
# Convention: branch name uses /, directory name uses -
git worktree add ../GUI_app-{safe-name} -b {branch/name}

# Examples:
git worktree add ../GUI_app-feature-cnn-fix -b feature/cnn-fix
git worktree add ../GUI_app-thesis-chapter-3 -b thesis/chapter-3
```
Then open the new directory in a separate VS Code window.

### List Worktrees
```powershell
git worktree list
```

### Merge Back to Main
```powershell
git checkout main
git merge {branch/name} --no-ff -m "Merge {branch/name}: {description}"
python -m pytest tests/ -v          # MUST pass before keeping merge
```

### Clean Up
```powershell
git worktree remove ../GUI_app-{safe-name}
git branch -d {branch/name}         # Only after successful merge
```

### PowerShell Helpers
Load with: `. .\.claude\scripts\worktree-helpers.ps1`
- `har-wt-new -BranchName "feature/cnn-fix"` — create
- `har-wt-list` — list all with status
- `har-wt-merge -BranchName "feature/cnn-fix"` — merge + test
- `har-wt-rm -BranchName "feature/cnn-fix" -DeleteBranch` — clean up

## Branch Naming
| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/{desc}` | `feature/cnn-generator-fix` |
| Bug fix | `fix/{desc}` | `fix/padding-bug` |
| Thesis | `thesis/{section}` | `thesis/chapter-3` |
| Experiment | `exp/{desc}` | `exp/new-feature-set` |

## Important Notes
- `persistent_data/` is gitignored — each worktree starts with empty runtime data
- Commits are visible across all worktrees (shared `.git`)
- Don't delete the main worktree
- Run tests after every merge
