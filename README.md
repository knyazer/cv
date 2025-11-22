# Very simple CV implemented in LaTeX

[Click here to view the pdf](./cv.pdf)

Self-contained, short, blue-ish.

* Not using fancy fonts

* Not using fancy images

* Not using fancy anything

Just a CV. Just some words.

```bash
sudo apt install texlive-full
sudo tlmgr init-usertree # on debian-based systems
sudo tlmgr install fontawesome5
pdflatex cv.tex && bibtex cv && pdflatex cv.tex
```

Or, to build it quickly you can use `just`


*Oh yeah, forgot to mention: this is actually my CV :)*

## How to use

Ask me, there is too much complex things involved

but like, approximately:
- put your openrouter key in `.env`
- write your own cv in the `/cv.tex`
- `rm -rf variants`

then you can run `uv run tailor.py` or `uv run tailor.py --write`, where the first one generates just the pdf in `/pdfs`, and the second one dumps stuff into `/variants` _and_ dumps the link to a pdf into `/pdfs`
