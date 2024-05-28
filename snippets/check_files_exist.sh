#!/usr/bin/env bash

set -euo pipefail

HAFH='/mnt/scratch-artemis/anilkeshwani'
filelist="${HAFH}/data/MLS/mls_english/train/transcripts_stratified_sample_2702009.list"
N=200

(
    cd "${HAFH}"/data/MLS
    count=0
    while IFS= read -r filepath; do
        if [ $count -lt $N ]; then
            if [ -f "$filepath" ]; then
                echo "File exists: $filepath"
            else
                echo "File does not exist: $filepath"
            fi
            count=$((count + 1))
        else
            break
        fi
    done <"$filelist"
)
