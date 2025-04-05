#!/bin/bash

# verify the rebuild
confirmation_message="The rebuild will delete and recompile all latex \
files. This may take a while. Are you sure you want to continue [Y/n]: "

read -p "$confirmation_message" response
response=${response,,} # cast to lower case
response=${response: -n} # default to no

if [[ "$response" == "y" ]]; then
    echo "Confirmation verified"
else
    echo "Cancelled"
    exit 0
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




# delete all pdfs in /pdf/
echo "Deleting all existing pdfs..."
find pdf/ -type f -name "*.pdf" -delete
echo "done"

# find all .tex files in /latex/ and build them
# tex_files=$(find latex/ -type f -name "*.tex")
# bash build.bash $tex_files
tex_files=()
while IFS= read -r tex; do
    if [[ "$tex" == "latex/school work/"* ]]; then
        # only include main.tex from this sub dir
        if [[ "$(basename "$tex")" == "main.tex" ]]; then
            tex_files+=("$tex")
        fi
    else
        # include all other .tex files
        tex_files+=("$tex")
    fi
    echo "FOUND: $tex"
done < <(find latex/ -type f -name "*.tex")
bash build.bash "${tex_files[@]}"
echo "Full rebuild done"