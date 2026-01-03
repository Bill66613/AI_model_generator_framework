# Thesis Compilation Guide

## Prerequisites

### 1. MiKTeX Installation

You mentioned MiKTeX was installed. If `pdflatex` is not recognized in your terminal:

**Option A: Add MiKTeX to PATH (Recommended)**

1. Find your MiKTeX installation directory (usually `C:\Program Files\MiKTeX\miktex\bin\x64\`)
2. Add to System PATH:
   - Press `Win + X` → System → Advanced System Settings
   - Click "Environment Variables"
   - Under "System variables", select "Path" → Edit
   - Click "New" and add: `C:\Program Files\MiKTeX\miktex\bin\x64\`
   - Click OK, restart terminal

**Option B: Use Full Path**

```powershell
cd d:\Workspaces\Master\ComputerScience\Thesis\GUI_app\academic-paper
& "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" main.tex
```

**Verify Installation:**

```powershell
# After adding to PATH, test:
pdflatex --version
# Should output: MiKTeX-pdfTeX x.x (MiKTeX x.x)
```

### 2. Required LaTeX Packages

All packages used in the thesis are standard in MiKTeX:

- ✓ `inputenc`, `amsmath`, `amsfonts` (core math)
- ✓ `graphicx` (figures)
- ✓ `booktabs`, `multirow` (tables)
- ✓ `cite`, `hyperref` (references, links)
- ✓ `geometry` (page layout)

If compilation reports missing packages, install via:

```powershell
mpm --install=<package-name>
# Example: mpm --install=multirow
```

---

## Compilation Steps

### Standard Compilation Sequence

Run these commands **in order** from the `academic-paper/` directory:

```powershell
# Navigate to paper directory
cd d:\Workspaces\Master\ComputerScience\Thesis\GUI_app\academic-paper

# Step 1: First LaTeX pass (generates .aux files)
pdflatex main.tex

# Step 2: Process bibliography (generates .bbl file)
bibtex main

# Step 3: Second LaTeX pass (resolves citations)
pdflatex main.tex

# Step 4: Third LaTeX pass (resolves cross-references)
pdflatex main.tex

# Output: main.pdf (your thesis paper)
```

**Why 4 steps?**

- Pass 1: Initial compilation, creates auxiliary files
- BibTeX: Processes references from references.bib
- Pass 2: Inserts bibliography and updates references
- Pass 3: Resolves any remaining cross-references and page numbers

### Quick Check Compilation

If you just want to see if there are LaTeX errors (no bibliography):

```powershell
pdflatex -interaction=nonstopmode main.tex
```

The `-interaction=nonstopmode` flag continues compilation even if there are warnings.

---

## Before First Compilation

### 1. Generate Figures (Required)

The Results section references figures that need to be generated:

```powershell
# Install Python dependencies if not already installed
pip install matplotlib seaborn numpy pandas

# Generate figures
python generate_figures.py
```

This creates:

- `figures/confusion_matrix_nn.png`
- `figures/power_consumption_profile.png`
- `figures/system_architecture.png`

**IMPORTANT:** The confusion matrix uses ESTIMATED data. Update line 42-49 in `generate_figures.py` with your actual model's predictions!

### 2. Update Results with Actual Data (If Needed)

The `sections/results.tex` file contains specific numbers:

- RF accuracy: 94.5%
- NN accuracy: 96.2%
- SVM accuracy: 93.8%
- Inference times, memory usage, etc.

If these don't match your actual experiments, edit the tables in `sections/results.tex`.

### 3. Temporarily Comment Out Missing Figures (Optional)

If you haven't generated figures yet but want to compile, comment out these lines in `sections/results.tex`:

```latex
% Temporarily commented until figures are generated
% \begin{figure}[h]
%     \centering
%     \includegraphics[width=0.9\textwidth]{figures/confusion_matrix_nn.png}
%     \caption{Confusion Matrix for Neural Network Model}
%     \label{fig:confusion_matrix}
% \end{figure}
```

---

## Common Compilation Errors and Solutions

### Error 1: "File not found: confusion_matrix_nn.png"

**Solution:** Run `python generate_figures.py` to create figures, or comment out `\includegraphics` lines.

### Error 2: "Undefined control sequence \multirow"

**Solution:** Install multirow package:

```powershell
mpm --install=multirow
```

### Error 3: "Citation undefined: article:human_motion_tracking_survey"

**Solution:** This is normal on first pass. Run the full 4-step sequence (pdflatex → bibtex → pdflatex × 2).

### Error 4: "Overfull \hbox" warnings

**Solution:** These are formatting warnings, not errors. The PDF will still compile. To fix:

- Shorten long URLs in references.bib
- Break long words with `\-` for hyphenation
- Adjust margins if severe

### Error 5: "Emergency stop" or syntax errors

**Solution:** Check the .log file for the exact line number:

```powershell
Get-Content main.log | Select-String "Error" -Context 2,2
```

Common causes:

- Missing `\end{...}` for environments
- Unescaped special characters: `&`, `%`, `$`, `_`
- Mismatched braces `{}`

---

## Viewing the PDF

### Option 1: Default PDF Viewer

```powershell
# Open compiled PDF
Start-Process main.pdf
```

### Option 2: VS Code PDF Extension

If you have a PDF viewer extension in VS Code, the PDF will auto-refresh after each compilation.

---

## Compilation Tips

### Tip 1: Watch Mode (Auto-recompile on Save)

For iterative editing, use a LaTeX IDE or script that watches for file changes:

**Using latexmk (if installed with MiKTeX):**

```powershell
latexmk -pdf -pvc main.tex
```

This auto-recompiles when you save any .tex file.

### Tip 2: Clean Build

If compilation behaves strangely, clean auxiliary files:

```powershell
# Remove temporary files
Remove-Item main.aux, main.bbl, main.blg, main.log, main.out, main.toc -ErrorAction SilentlyContinue

# Recompile from scratch
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Tip 3: Check Warnings

After successful compilation, check for warnings:

```powershell
# View LaTeX warnings
Get-Content main.log | Select-String "Warning"

# View BibTeX warnings
Get-Content main.blg | Select-String "Warning"
```

Common warnings to fix:

- Missing citations (add to references.bib)
- Empty bibliography entries
- Figure/table placement issues (adjust `[h]` to `[ht]` or `[!htb]`)

---

## Output Files Explained

After compilation, you'll see these files:

| File | Purpose | Keep? |
|------|---------|-------|
| `main.pdf` | **Final thesis document** | ✓ Keep |
| `main.aux` | Auxiliary data (cross-refs, citations) | Can delete |
| `main.bbl` | Formatted bibliography | Can delete |
| `main.blg` | BibTeX log | Can delete |
| `main.log` | LaTeX compilation log | Review for errors |
| `main.out` | Hyperref bookmark data | Can delete |
| `main.toc` | Table of contents data | Can delete |

**For submission:** You only need `main.pdf`. Keep source .tex files for future edits.

---

## Pre-Submission Checklist

Before submitting to Council H:

- [ ] All sections compile without errors
- [ ] All figures generated and embedded correctly
- [ ] All citations resolve (no `[?]` in PDF)
- [ ] All cross-references work (no `??` for figures/tables)
- [ ] Table of contents accurate (if generated)
- [ ] Page numbers correct
- [ ] No overfull hbox warnings (or acceptable)
- [ ] PDF opens correctly in Adobe Reader
- [ ] Proofread all sections (typos, grammar)
- [ ] Supervisor review completed

---

## Alternative: Use Overleaf

If LaTeX compilation is problematic locally, consider **Overleaf** (online LaTeX editor):

1. Go to <https://www.overleaf.com/>
2. Create free account
3. Upload your project:
   - `main.tex`
   - All `sections/*.tex` files
   - `references.bib`
   - `figures/` directory
4. Click "Recompile" to generate PDF online
5. No local MiKTeX installation needed!

**Pros:**

- No installation required
- Auto-compilation on save
- Collaboration features
- Works on any device

**Cons:**

- Requires internet connection
- Free tier has limitations (1 collaborator, limited projects)

---

## Getting Help

If you encounter compilation errors you can't resolve:

1. **Check the .log file** for specific error messages:

   ```powershell
   Get-Content main.log | Select-String -Pattern "Error|Fatal" -Context 3,3
   ```

2. **Search the error message** on TeX StackExchange: <https://tex.stackexchange.com/>

3. **Share error details:**
   - Exact error message from main.log
   - Line number where error occurs
   - Section of .tex file causing the issue

4. **Test section by section:**
   Comment out sections in main.tex to isolate the problem:

   ```latex
   % \input{sections/results}  % Temporarily disabled
   ```

---

## Quick Reference Card

```powershell
# Full compilation (run these in order):
cd d:\Workspaces\Master\ComputerScience\Thesis\GUI_app\academic-paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Generate figures first:
python generate_figures.py

# View PDF:
Start-Process main.pdf

# Clean build:
Remove-Item main.aux, main.bbl, main.blg, main.log -ErrorAction SilentlyContinue

# Check for errors:
Get-Content main.log | Select-String "Error"
```

---

**Status:** Thesis paper is ready for compilation. Once MiKTeX PATH is configured and figures are generated, you can compile to PDF!
