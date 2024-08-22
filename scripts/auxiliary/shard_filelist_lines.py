#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import shard_jsonl


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--n-shards", type=int)
    parser.add_argument("--shard_dir", type=Path, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    shard_jsonl(**vars(args))
