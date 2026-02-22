# 11. PDF EXPORT INSTRUCTIONS

## Pandoc Export Configuration

To export this course as a professional PDF using Pandoc:

**Install Pandoc**:
```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended

# Windows
# Download from: https://pandoc.org/installing.html
```

**Export Command**:
```bash
pandoc RAG_KG_Master_Course.md \
  -o RAG_KG_Master_Course.pdf \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=3 \
  --number-sections \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=report \
  -V linkcolor=blue \
  -V urlcolor=blue
```

## PDF-Optimized Version

Create a file `RAG_KG_Master_Course_PDF.md` with YAML frontmatter:

```yaml
---
title: "RAG + Knowledge Graph Master Course"
subtitle: "From Beginner to Hire-Ready Enterprise AI Engineer"
author: "AI Engineering Academy"
date: "2026"
toc: true
toc-depth: 3
numbersections: true
geometry: margin=1in
fontsize: 11pt
documentclass: report
linkcolor: blue
urlcolor: blue
---
```

Then run:
```bash
pandoc RAG_KG_Master_Course_PDF.md RAG_KG_Master_Course.md \
  -o RAG_KG_Master_Course.pdf \
  --pdf-engine=xelatex
```

## Alternative: HTML Export

For a web-friendly version:

```bash
pandoc RAG_KG_Master_Course.md \
  -o RAG_KG_Master_Course.html \
  --standalone \
  --toc \
  --toc-depth=3 \
  --css=style.css \
  --metadata title="RAG + KG Master Course"
```

---
