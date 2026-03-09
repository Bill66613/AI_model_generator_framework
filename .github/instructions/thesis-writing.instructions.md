---
description: "Vietnamese thesis writing conventions, LaTeX compilation, and academic terminology for thesis chapters and sections."
applyTo: "academic-paper-vietnamese/**"
---
# Thesis Writing Rules

## Language
All thesis content must be in **Vietnamese**. Use formal academic register.

## Compilation
```bash
cd academic-paper-vietnamese
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Documentation chain
- `TECHNICAL_FINDINGS.md` — Detailed technical evidence (11 findings)
- `THESIS_REPORT_INSTRUCTIONS.md` — TODO list, chapter mapping, terminology glossary

## Rules
- Every technical claim must reference code or data evidence
- Cross-reference findings by number (e.g., "Finding §7: Feature order mismatch")
- Use the Vietnamese terminology glossary in THESIS_REPORT_INSTRUCTIONS.md
