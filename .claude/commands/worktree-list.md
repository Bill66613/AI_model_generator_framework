# Git Worktree — List and manage existing worktrees

List all active worktrees and optionally clean up completed ones.

## Steps

1. List all worktrees:
```bash
git worktree list
```

2. For each worktree, show its branch status:
```bash
git worktree list --porcelain
```

3. Check if any worktrees have unmerged changes:
```bash
for worktree in $(git worktree list --porcelain | grep "^worktree" | awk '{print $2}'); do
  echo "=== $worktree ==="
  git -C "$worktree" status --short
  git -C "$worktree" log --oneline -3
done
```

## Cleanup
To remove a completed worktree:
```bash
git worktree remove ../GUI_app-{branch-name}
# Optionally delete the branch too:
git branch -d {branch-name}
```

To prune stale worktree references:
```bash
git worktree prune
```

$ARGUMENTS
