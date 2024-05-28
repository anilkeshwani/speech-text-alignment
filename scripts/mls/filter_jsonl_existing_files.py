#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import mls_id_to_path, read_jsonl, write_jsonl


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
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--suffix", type=str, default=".flac")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.jsonl.with_stem(args.jsonl.stem + "_existing_files_only")
    return args


def main(jsonl: Path, out: Path, audio_dir: Path, head: int | None, suffix: str):
    dataset = read_jsonl(jsonl)
    filtered_dataset: list[dict] = []
    for i, sample in enumerate(dataset):
        if head is not None and i == head:
            break
        else:
            if mls_id_to_path(sample["ID"], audio_dir, suffix=suffix).exists():
                filtered_dataset.append(sample)
    write_jsonl(out, filtered_dataset)
    LOGGER.info(f"Wrote filtered JSON lines containing only existing files to {out!s}")


if __name__ == "__main__":
    main(**vars(parse_args()))
