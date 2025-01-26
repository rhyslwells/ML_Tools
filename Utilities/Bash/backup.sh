#!/bin/bash
source_dir=$1
backup_dir=$2
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

if [ -d "$source_dir" ]; then
    mkdir -p "$backup_dir"
    tar -czf "$backup_dir/backup_$timestamp.tar.gz" "$source_dir"
    echo "Backup created at $backup_dir/backup_$timestamp.tar.gz"
else
    echo "Source directory $source_dir does not exist."
fi

# Usage: ./backup.sh /path/to/source /path/to/backup
# ./backup.sh \test_folder endpoint
#Back up a directory to a specified location with a timestamp.