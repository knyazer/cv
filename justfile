default: build

build:
    pdflatex -interaction=nonstopmode -halt-on-error cv.tex
    bibtex cv
    pdflatex -interaction=nonstopmode -halt-on-error cv.tex

