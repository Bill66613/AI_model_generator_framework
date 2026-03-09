---
description: "Create a new git worktree for parallel Claude Code or Copilot sessions on separate branches"
agent: "agent"
argument-hint: "Branch name (e.g., feature/cnn-fix, thesis/chapter-3)"
---
Create a new git worktree for parallel work. Each worktree gets its own directory so separate agent sessions don't block each other.

## Naming Convention
- Feature work: `feature/{short-description}`
- Bug fixes: `fix/{short-description}`
- Thesis work: `thesis/{chapter-or-section}`
- Experiments: `exp/{short-description}`

## Steps
1. Check current status: `git status` and `git worktree list`
2. Create worktree (replace `$BRANCH` with the branch name, `$SAFE` with branch name using `-` instead of `/`):
   ```bash
   git worktree add ../GUI_app-$SAFE -b $BRANCH
   ```
3. Confirm: `git worktree list`
4. Tell the user to open the new worktree path in a separate VS Code window.

## Important
- Worktrees share `.git` — commits are visible across all
- `persistent_data/` is gitignored — each worktree starts with empty runtime data
- Clean up with: `git worktree remove ../GUI_app-$SAFE`
