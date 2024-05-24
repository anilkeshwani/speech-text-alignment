#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sardalign.constants import SEED
from sardalign.utils import count_lines, get_integer_sample_size, parse_arg_int_or_float
from sardalign.utils.data import get_stratified_sample


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--transcripts-jsonl", required=True, type=Path, help="transcripts.jsonl")
    parser.add_argument("--output-jsonl", default=None, type=Path, help="Output path for stratified sample jsonl")
    parser.add_argument("--sample", required=True, type=parse_arg_int_or_float, help="transcripts.txt")
    parser.add_argument("--force-ascii", action="store_true", help="Escape non-ASCII characters in JSON lines output")
    parser.add_argument("--shuffle", action="store_true", help="Whether to randomise the sample from each stratum")
    parser.add_argument("--seed", default=SEED, type=int, help="Random seed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    N = count_lines(args.transcripts_jsonl)
    sample_size = get_integer_sample_size(args.sample, N)
    if args.output_jsonl is None:
        stem = args.transcripts_jsonl.stem + f"_stratified_sample_{sample_size}"
        args.output_jsonl = args.transcripts_jsonl.with_stem(stem)
    if args.output_jsonl.exists():
        raise FileExistsError(f"File already present at {args.output_jsonl}")
    # transcripts JSON lines contains an "ID" column of the form 4800_10003_000000 which is {speaker}_{book}_{audio}
    transcripts = pd.read_json(args.transcripts_jsonl, lines=True, orient="records", dtype={"ID": str})
    transcripts["speaker"] = transcripts["ID"].str.split("_").str[0].astype(int)  # speaker ID from ID column
    stratified_sample, _N = get_stratified_sample(
        transcripts,
        sample_size,
        strata_label="speaker",
        shuffle=args.shuffle,
        verbose=args.verbose,
        seed=args.seed,
        logger=LOGGER,
    )
    assert N == _N
    stratified_sample.drop(columns="speaker", inplace=True)
    if args.verbose:
        LOGGER.info(stratified_sample.head())
    with open(args.output_jsonl, "x") as f:
        f.write(stratified_sample.to_json(orient="records", lines=True, force_ascii=args.force_ascii))
    LOGGER.info(f"Wrote {sample_size} lines to {args.output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
