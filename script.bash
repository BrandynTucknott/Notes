#!/bin/bash

# compile latex/test/hello.tex into a pdf and place it into pdf/test
pdflatex latex/test/hello.tex
mv hello.pdf pdf/test
rm hello.aux hello.log hello.out