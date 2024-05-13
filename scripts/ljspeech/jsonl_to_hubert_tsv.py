#!/usr/bin/env python

import argparse
from pathlib import Path

from sardalign.utils import read_jsonl


def main(args):
    data = read_jsonl(args.in_jsonl_path)
    wavs: list[str] = [f"{s['ID']}.wav" for s in data]
    lines = [str(args.ljspeech_wavs_dir)] + wavs
    with open(args.out_tsv_path, "x") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LJSpeech metadata from JSONL to TSV format.")
    parser.add_argument("--in-jsonl-path", type=Path, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--out-tsv-path", type=Path, required=True, help="Path to the output TSV file.")
    parser.add_argument("--ljspeech-wavs-dir", type=Path, required=True, help="Path to the LJSpeech WAVs directory.")

    args = parser.parse_args()
    main(args)
