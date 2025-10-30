#!/bin/bash

# DELETE after script completion
echo "script incomplete, failing without action."
exit 0

# template path from repo home
template_path="lib/template.tex"
create_path=false

# any arguments provided?
if [ -z "$1" ]; then
    echo "Usage failed: no arguments provided"
    echo "Usage: bash create.bash <path> [--create-path=true|false]"
    exit 1
fi

# more than one argument given, fail without action
if (( $# > 2 )); then
    echo "Usage failed: more than two arguments given"
    echo "Usage: bash create.bash <path> [--create-path=true|false]"
    exit 1
fi

path="$1"

# parse flags
for arg in "$@"; do
    case $arg in 
        --create-path=true)
        create_path=true
        ;;
        --create-path=false)
        create_path=false
        ;;
    esac
done

# we hope the given argument is a path
