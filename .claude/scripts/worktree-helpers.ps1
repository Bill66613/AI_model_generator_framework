# HAR Framework — Git Worktree Helper Scripts (PowerShell)
# Usage: . .\.claude\scripts\worktree-helpers.ps1

function New-HARWorktree {
    <#
    .SYNOPSIS
    Create a new git worktree for parallel Claude Code sessions.
    
    .PARAMETER BranchName
    Branch name (e.g., feature/cnn-fix, thesis/chapter-3)
    
    .PARAMETER BaseBranch
    Base branch to create from (default: main)
    
    .EXAMPLE
    New-HARWorktree -BranchName "feature/cnn-fix"
    New-HARWorktree -BranchName "thesis/chapter-3" -BaseBranch "main"
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$BranchName,
        [string]$BaseBranch = "main"
    )
    
    $SafeName = $BranchName -replace '/', '-'
    $WorktreePath = "../GUI_app-$SafeName"
    
    if (Test-Path $WorktreePath) {
        Write-Host "❌ Worktree already exists at $WorktreePath" -ForegroundColor Red
        return
    }
    
    Write-Host "🌿 Creating worktree for branch '$BranchName' at $WorktreePath..." -ForegroundColor Cyan
    git worktree add $WorktreePath -b $BranchName $BaseBranch
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Worktree created successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Open new VS Code window: code $WorktreePath"
        Write-Host "  2. Start Claude Code in that window"
        Write-Host "  3. Each session works independently on its branch"
        Write-Host ""
        Write-Host "To merge back: /worktree-merge $BranchName"
    } else {
        Write-Host "❌ Failed to create worktree" -ForegroundColor Red
    }
}

function Get-HARWorktrees {
    <#
    .SYNOPSIS
    List all active worktrees with status.
    #>
    Write-Host "🌿 Active Worktrees:" -ForegroundColor Cyan
    Write-Host "===================" -ForegroundColor Cyan
    
    $worktrees = git worktree list
    foreach ($wt in $worktrees) {
        Write-Host $wt
        
        # Extract path
        $path = ($wt -split '\s+')[0]
        if (Test-Path $path) {
            $changes = (git -C $path status --short | Measure-Object).Count
            if ($changes -gt 0) {
                Write-Host "  ⚠️  $changes uncommitted changes" -ForegroundColor Yellow
            } else {
                Write-Host "  ✅ Clean" -ForegroundColor Green
            }
        }
    }
}

function Remove-HARWorktree {
    <#
    .SYNOPSIS
    Remove a completed worktree and optionally delete its branch.
    
    .PARAMETER BranchName
    Branch name to remove
    
    .PARAMETER DeleteBranch
    Also delete the git branch (default: false)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$BranchName,
        [switch]$DeleteBranch
    )
    
    $SafeName = $BranchName -replace '/', '-'
    $WorktreePath = "../GUI_app-$SafeName"
    
    if (-not (Test-Path $WorktreePath)) {
        Write-Host "❌ Worktree not found at $WorktreePath" -ForegroundColor Red
        return
    }
    
    # Check for uncommitted changes
    $changes = (git -C $WorktreePath status --short | Measure-Object).Count
    if ($changes -gt 0) {
        Write-Host "⚠️  Worktree has $changes uncommitted changes!" -ForegroundColor Yellow
        $confirm = Read-Host "Are you sure you want to remove it? (y/N)"
        if ($confirm -ne 'y') {
            Write-Host "Cancelled." -ForegroundColor Yellow
            return
        }
    }
    
    Write-Host "🗑️  Removing worktree at $WorktreePath..." -ForegroundColor Cyan
    git worktree remove $WorktreePath
    
    if ($DeleteBranch) {
        Write-Host "🗑️  Deleting branch $BranchName..." -ForegroundColor Cyan
        git branch -d $BranchName
    }
    
    Write-Host "✅ Done." -ForegroundColor Green
}

function Merge-HARWorktree {
    <#
    .SYNOPSIS
    Merge a worktree branch back into main.
    
    .PARAMETER BranchName
    Branch name to merge
    
    .PARAMETER Description
    One-line description for the merge commit
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$BranchName,
        [string]$Description = "Merge $BranchName"
    )
    
    # Save current branch
    $currentBranch = git branch --show-current
    
    Write-Host "🔀 Merging $BranchName into main..." -ForegroundColor Cyan
    
    # Switch to main
    git checkout main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to checkout main" -ForegroundColor Red
        return
    }
    
    # Merge
    git merge $BranchName --no-ff -m "Merge $BranchName`: $Description"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Merge conflicts! Resolve manually." -ForegroundColor Red
        return
    }
    
    # Run tests
    Write-Host "🧪 Running tests..." -ForegroundColor Cyan
    python -m pytest tests/ -v --tb=short
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Tests failed after merge! Consider reverting." -ForegroundColor Yellow
        return
    }
    
    Write-Host "✅ Merge successful and tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next: Remove-HARWorktree -BranchName '$BranchName' -DeleteBranch" -ForegroundColor Yellow
}

# Quick aliases
Set-Alias -Name har-wt-new -Value New-HARWorktree
Set-Alias -Name har-wt-list -Value Get-HARWorktrees
Set-Alias -Name har-wt-rm -Value Remove-HARWorktree
Set-Alias -Name har-wt-merge -Value Merge-HARWorktree

Write-Host "🔧 HAR Worktree helpers loaded!" -ForegroundColor Green
Write-Host "   Commands: New-HARWorktree, Get-HARWorktrees, Remove-HARWorktree, Merge-HARWorktree" -ForegroundColor Gray
Write-Host "   Aliases:  har-wt-new, har-wt-list, har-wt-rm, har-wt-merge" -ForegroundColor Gray
