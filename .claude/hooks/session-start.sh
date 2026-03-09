#!/bin/bash
# Session start hook
# Load context when a new Claude Code session begins

echo "📂 HAR Edge Deployment Framework"
echo "================================"

# Show current branch and worktree
echo ""
echo "🌿 Git Status:"
git branch --show-current
git worktree list 2>/dev/null

# Show recent commits
echo ""
echo "📝 Recent commits:"
git log --oneline -5

# Check for session notes
echo ""
if [ -d ".claude/sessions" ] && [ "$(ls -A .claude/sessions 2>/dev/null)" ]; then
  LATEST=$(ls -t .claude/sessions/*.md 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    echo "📋 Latest session note: $LATEST"
    echo "   Use /resume to pick up from last session"
  fi
else
  echo "📋 No previous session notes found."
fi

# Check for uncommitted changes
CHANGES=$(git status --short | wc -l)
if [ "$CHANGES" -gt 0 ]; then
  echo ""
  echo "⚠️  $CHANGES uncommitted changes detected"
  git status --short
fi

echo ""
echo "Available commands: /thesis /codegen /frontend /deploy /review /test"
echo "Worktree commands: /worktree-create /worktree-list /worktree-merge"
echo "Session commands:  /handoff /resume"
