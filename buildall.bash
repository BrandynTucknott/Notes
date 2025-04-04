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

# delete all pdfs in pdf/
# find all .tex files in latex/
# call build.bash will all .tex files