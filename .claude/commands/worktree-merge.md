# Merge worktree branch back to main

Merge a completed feature branch from a worktree back into the main branch.

## Steps

1. List worktrees to identify the branch:
```bash
git worktree list
```

2. Ensure the feature branch is clean:
```bash
git -C ../GUI_app-$BRANCH_NAME status
```

3. Switch to main and merge:
```bash
git checkout main
git merge $BRANCH_NAME --no-ff -m "Merge $BRANCH_NAME: $DESCRIPTION"
```

4. Run tests after merge:
```bash
python -m pytest tests/ -v
```

5. Clean up the worktree:
```bash
git worktree remove ../GUI_app-$BRANCH_NAME
git branch -d $BRANCH_NAME
```

## Rules
- Always use `--no-ff` to preserve branch history
- Run tests after merge
- If conflicts exist, resolve them manually and document what was changed
- If the merge involves code generator changes, run the parity review: `/review parity check on merged changes`

$ARGUMENTS
