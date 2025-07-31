#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.data.mls import mls_id_to_path
from sardalign.utils import write_jsonl


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None, help="Maximum number of existing samples to take")
    parser.add_argument("--suffix", type=str, default=".flac")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing output file")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.jsonl.with_stem(args.jsonl.stem + "_existing_files_only")
    return args


def main(jsonl: Path, out: Path, audio_dir: Path, n: int | None, suffix: str, overwrite: bool):
    filtered_dataset: list[dict] = []
    with open(jsonl) as f:
        for line in f:
            sample = json.loads(line)
            if len(filtered_dataset) == n:
                break
            if mls_id_to_path(sample["ID"], audio_dir, suffix=suffix).exists():
                filtered_dataset.append(sample)
    write_jsonl(out, filtered_dataset, mode="w" if overwrite else "x")
    LOGGER.info(f"Wrote filtered JSON lines containing {len(filtered_dataset)} existing files to {out!s}")


if __name__ == "__main__":
    main(**vars(parse_args()))
