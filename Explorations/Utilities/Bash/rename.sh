#!/bin/bash
dir=$1
prefix=$2

if [ -d "$dir" ]; then
    for file in "$dir"/*; do
        mv "$file" "$dir/$prefix$(basename "$file")"
    done
    echo "Files renamed with prefix '$prefix'."
else
    echo "Directory $dir does not exist."
fi

#Rename all files in a directory by appending a prefix.
# ./rename.sh /path/to/directory prefix_
# ./rename.sh \Renaming_folder words