# Git Worktree — Create a new worktree for parallel work

Create a new git worktree so a separate Claude Code session can work on a feature branch without blocking this session.

## Steps

1. First, check current branch and status:
```bash
git status
git branch -a
git worktree list
```

2. Create branch and worktree. The branch name should describe the work (e.g., `feature/cnn-generator-fix`, `thesis/chapter-3`):
```bash
git worktree add ../GUI_app-$BRANCH_NAME -b $BRANCH_NAME
```

3. Confirm the worktree was created:
```bash
git worktree list
```

4. Report the worktree path to the user so they can open it in a new VS Code window / Claude Code session.

## Naming Convention
- Feature work: `feature/{short-description}`
- Bug fixes: `fix/{short-description}`
- Thesis work: `thesis/{chapter-or-section}`
- Experiments: `exp/{short-description}`

## Worktree Location
All worktrees go in the parent directory: `../GUI_app-{branch-name}/`

## Important
- Each worktree shares the same `.git` — commits are visible across all worktrees
- Don't delete the main worktree
- Clean up finished worktrees with: `git worktree remove ../GUI_app-{branch-name}`
- `persistent_data/` is gitignored — each worktree starts with empty runtime data

$ARGUMENTS
