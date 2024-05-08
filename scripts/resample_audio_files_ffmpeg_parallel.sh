#!/usr/bin/env bash

IN_DIR='/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs'
OUT_DIR='/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs_16000_7'
TARGET_SR=16000
N_JOBS=32

# Ensure the target directory exists
mkdir -p "$OUT_DIR"

# Find.wav files and process them in parallel
find "$IN_DIR" -type f -name "*.wav" | parallel "-j${N_JOBS}" "ffmpeg -i {} -ar ${TARGET_SR} ${OUT_DIR}/{/}"
