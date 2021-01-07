pandoc -o docs.tex quickintro.md benchmark.md  ../README.md -s
pdflatex docs.tex
# \DeclareUnicodeCharacter{2500}{─}
# \DeclareUnicodeCharacter{03F5}{ϵ}
# pandoc -o docs.html quickintro.md benchmark.md  ../README.md
