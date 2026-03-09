# Claude Code Workflow Orchestration

## Quick Start

```bash
# Install Claude Code (if not already)
npm install -g @anthropic-ai/claude-code

# Start Claude Code in this project
claude

# Load worktree helpers for PowerShell
. .\.claude\scripts\worktree-helpers.ps1
```

## Slash Commands

| Command | Purpose | Agent Specialty |
|---------|---------|-----------------|
| `/thesis` | Write/edit thesis LaTeX chapters | Vietnamese academic writing, LaTeX |
| `/codegen` | Modify code generators | Training-deployment parity (CRITICAL) |
| `/frontend` | Dash layouts & callbacks | UI/UX, Dash components |
| `/deploy` | Device deployment & testing | Embedded C++, serial communication |
| `/review` | Code review with parity checks | Quality assurance |
| `/test` | Run tests | pytest, validation |
| `/status` | Quick project health check | Git, tests, TODOs |
| `/handoff` | Save session context | Cross-session continuity |
| `/resume` | Resume from last session | Context loading |
| `/worktree-create` | New worktree for parallel work | Git worktrees |
| `/worktree-list` | List active worktrees | Git worktrees |
| `/worktree-merge` | Merge branch back to main | Git, testing |

## Multi-Session Parallel Work with Git Worktrees

### The Problem
A single Claude Code session blocks on one task. If you're doing thesis writing, you can't simultaneously fix a code generator bug.

### The Solution: Git Worktrees
Each Claude Code session gets its own worktree (separate working directory, same git repo):

```
D:\...\GUI_app\              ← main branch (Session 1: thesis writing)
D:\...\GUI_app-feature-cnn\  ← feature/cnn-fix (Session 2: code gen work)
D:\...\GUI_app-fix-padding\  ← fix/padding-bug (Session 3: bugfix)
```

### Workflow

**Session 1** — Start thesis work:
```
claude> /worktree-create thesis/chapter-3
# Opens new worktree at ../GUI_app-thesis-chapter-3/
```

**Session 2** — In a new VS Code window, open the worktree:
```bash
code ../GUI_app-thesis-chapter-3/
# Start claude in that window
claude> /thesis Write section 3.2 on feature extraction
```

**Meanwhile, Session 1** continues with code generation work on main.

**When done** — Merge back:
```
claude> /worktree-merge thesis/chapter-3
```

### PowerShell Helpers
```powershell
# Load helpers
. .\.claude\scripts\worktree-helpers.ps1

# Create worktree
har-wt-new -BranchName "feature/cnn-fix"

# List all worktrees
har-wt-list

# Merge and clean up
har-wt-merge -BranchName "feature/cnn-fix" -Description "Fixed CNN generator parity"
har-wt-rm -BranchName "feature/cnn-fix" -DeleteBranch
```

## Hooks

| Hook | Trigger | Purpose |
|------|---------|---------|
| `pre-edit-parity-check` | Before editing parity-critical files | Warns about parity requirements |
| `post-edit-reminder` | After editing any code | Reminds about testing & sync |
| `pre-commit-checks` | Before git commit | Runs tests, checks for common issues |
| `session-start` | When Claude Code starts | Loads context, shows status |

## Session Continuity

### Saving Context (End of Session)
```
claude> /handoff
```
Creates a session note in `.claude/sessions/` with:
- What was accomplished
- In-progress work
- Blockers and decisions needed
- Files changed
- Next steps

### Resuming (Start of Session)
```
claude> /resume
```
Reads latest session note and project status files.

## File Structure

```
.claude/
├── settings.json              ← Permissions & environment
├── hooks.json                 ← Hook configuration
├── commands/                  ← Slash commands (agent specialties)
│   ├── thesis.md             ← /thesis — LaTeX writing
│   ├── codegen.md            ← /codegen — Code generators
│   ├── frontend.md           ← /frontend — Dash UI
│   ├── deploy.md             ← /deploy — Device deployment
│   ├── review.md             ← /review — Code review
│   ├── test.md               ← /test — Test runner
│   ├── status.md             ← /status — Health check
│   ├── handoff.md            ← /handoff — Session save
│   ├── resume.md             ← /resume — Session load
│   ├── worktree-create.md    ← /worktree-create
│   ├── worktree-list.md      ← /worktree-list
│   └── worktree-merge.md     ← /worktree-merge
├── hooks/                     ← Hook scripts
│   ├── pre-edit-parity-check.sh
│   ├── post-edit-reminder.sh
│   ├── pre-commit-checks.sh
│   └── session-start.sh
├── scripts/                   ← PowerShell helpers
│   └── worktree-helpers.ps1
└── sessions/                  ← Session handoff notes (gitignored)
    └── README.md
CLAUDE.md                      ← Main project rules & context
```

## Team Workflow Example

A typical thesis development day might look like:

1. **Morning** — Start Claude Code on `main`, run `/resume` to check status
2. **Task 1** — `/worktree-create feature/sliding-window` → work on feature in Session 2
3. **Task 2** — `/thesis Chapter 4 results section` → thesis writing in Session 1
4. **Midday** — Session 2 finishes → `/worktree-merge feature/sliding-window`
5. **Task 3** — `/codegen Add confidence threshold to SVM generator` → code gen work
6. **End of day** — `/handoff` in each active session to save context
