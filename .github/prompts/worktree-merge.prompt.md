---
description: "Merge a feature branch from a worktree back into main with tests"
agent: "agent"
argument-hint: "Branch name to merge (e.g., feature/cnn-fix)"
---
Merge a completed feature branch back into main:

1. List worktrees: `git worktree list`
2. Ensure the branch is clean: `git -C ../GUI_app-$SAFE status`
3. Switch to main and merge:
   ```bash
   git checkout main
   git merge $BRANCH --no-ff -m "Merge $BRANCH: $DESCRIPTION"
   ```
4. Run tests: `python -m pytest tests/ -v`
5. If tests pass, clean up:
   ```bash
   git worktree remove ../GUI_app-$SAFE
   git branch -d $BRANCH
   ```
6. If code generator files were changed, also run `/review` on the merged changes.
