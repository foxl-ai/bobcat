#!/usr/bin/env bash
# Build the Bobcat paper PDF and an arXiv source tarball. Nothing is uploaded or submitted.
# Run on the dev node (not the owner's Mac): bash scripts/build_arxiv.sh
# Output: paper/build/main.pdf and paper/build/arxiv-bobcat.tar.gz (main.tex, sections/,
# figures/*.pdf, refs.bib and the main.bbl that arXiv needs, since it does not run BibTeX).
set -euo pipefail
cd "$(dirname "$0")/../paper"
rm -rf build && mkdir -p build
latex() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex > /dev/null; }
latex
BIBINPUTS=. bibtex build/main > build/bibtex.out
latex
for _ in 1 2 3; do
  latex
  grep -q "Rerun to get" build/main.log || break
done
if grep -n '\\todo{' main.tex sections/*.tex | grep -v -E 'DeclareRobustCommand|^[^:]+:[0-9]+: *%'; then
  echo "Unfinished \\todo markers remain." >&2; exit 1
fi
if grep -E "undefined|Citation .* undefined|There were undefined" build/main.log; then
  echo "Undefined references or citations." >&2; exit 1
fi
stage=$(mktemp -d)
cp -r main.tex refs.bib sections "$stage/"
mkdir -p "$stage/figures" && cp figures/*.pdf "$stage/figures/"
cp build/main.bbl "$stage/main.bbl"
tar -czf build/arxiv-bobcat.tar.gz -C "$stage" .
rm -rf "$stage"
pages=$(grep -o 'Output written on build/main.pdf ([0-9]* pages' build/main.log | grep -o '[0-9]*' | tail -1 || true)
echo "built paper/build/main.pdf (${pages:-?} pages) and paper/build/arxiv-bobcat.tar.gz; not submitted"
