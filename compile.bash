#!/bin/bash

# is provided file a .tex file?
if [ -z "$1" ]; then
    echo "Usage failed: no arguments provided"
    exit 1
elif [[ "$1" != *.tex ]]; then
    echo "Usage failed: provided file must be a .tex file"
    exit 1
fi

# get top level dir of git repo
repo_root=$(git rev-parse --show-toplevel 2>/dev/null)

# are we in a git repo?
if [ -z "$repo_root" ]; then
    echo "Usage failed: not in git repo"
    exit 1
fi

# get current dir relative to git root
curr_path=$(pwd | sed "s|$repo_root/||")

# change dir to git root or exit if fail
cd "$repo_root" || exit 1




# latex_path="$curr_path"

# find path for .tex file
file="$1"
# path="${curr_path#latex/}" # latex/PATH/file.tex --> pdf/PATH

fname=$(basename "$file" .tex)

latex_path="$curr_path/$file"
latex_path="${latex_path%/$fname.tex}"
latex_path="${latex_path#$(pwd)/}"


pdf_path="pdf/${latex_path#latex/}"

# echo "latex path: $latex_path"
# echo "pdf path: $pdf_path"
# exit

# # compile .tex --> .pdf
pdflatex $latex_path/$fname.tex


# # delete extra .aux .log .out
$(rm $fname.aux $fname.log $fname.out)


# # create necessary directories if they do not exist in latex
$(mkdir -p $pdf_path)


# # move compiled .pdf file to pdf dir via absolute path
$(mv "$fname.pdf" "$pdf_path")

echo "dirname results: $dir"
echo "pdf path: $pdf_path"
echo "file name: $fname"
