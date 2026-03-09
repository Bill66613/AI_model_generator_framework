# Session Handoff — Save context for the next session

Save the current session's context so the next Claude Code session (or a different worktree session) can pick up where you left off.

## Steps

1. Summarize what was accomplished in this session
2. Note any in-progress work
3. List any blockers or decisions needed
4. Update relevant tracking files:

```
# Update TECHNICAL_FINDINGS.md if new findings were made
academic-paper-vietnamese/TECHNICAL_FINDINGS.md

# Update thesis instructions if thesis work was done
academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md

# Update TODO.md with current status
TODO.md
```

5. Create a session note at `.claude/sessions/` with the format:
```
.claude/sessions/YYYY-MM-DD-{topic}.md
```

## Session Note Template
```markdown
# Session: {date} — {topic}

## Accomplished
- ...

## In Progress
- ...

## Blockers / Decisions Needed
- ...

## Files Changed
- ...

## Next Steps
- ...

## Worktree
- Branch: {branch-name}
- Worktree path: {path}
```

$ARGUMENTS
