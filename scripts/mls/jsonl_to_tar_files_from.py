#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import mls_id_to_path, read_jsonl


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--transcripts", type=Path, help="Path to the MLS transcripts.txt file.")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Audio directory without leading slash / in MLS tar.gz file e.g. 'mls_english/train/audio/'",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to the output JSON lines file.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.transcripts.with_suffix(".list")
    return args


def main(args):
    args.output.write_text(
        "\n".join([str(mls_id_to_path(s["ID"], args.audio_dir)) for s in read_jsonl(args.transcripts)]) + "\n"
    )
    LOGGER.info(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main(parse_args())
