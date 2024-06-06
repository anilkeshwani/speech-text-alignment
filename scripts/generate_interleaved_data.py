#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import HUBERT_DOWNSAMPLING_RATIO, SAMPLING_FREQ
from sardalign.utils import count_lines, read_jsonl
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input-jsonl", type=Path, help="Path to jsonl with text alignments and HuBERT speech tokens")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to write interleaved data")
    args = parser.parse_args()
    return args


def main(args: Namespace):
    dataset = read_jsonl(args.input_jsonl)
    for sample in dataset:
        
        start_sp_token = int(span_start_sec * SAMPLING_FREQ / HUBERT_DOWNSAMPLING_RATIO)
        end_sp_token = int(ceil(span_end_sec * SAMPLING_FREQ / HUBERT_DOWNSAMPLING_RATIO))
