#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import pandas as pd
from sardalign.constants import SEED
from sardalign.utils import parse_arg_int_or_float
from sklearn.model_selection import train_test_split


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--transcripts-jsonl", required=True, type=Path, help="transcripts.jsonl")
    parser.add_argument("--output-jsonl", default=None, type=Path, help="Output path for stratified sample jsonl")
    parser.add_argument("--sample", required=True, type=parse_arg_int_or_float, help="transcripts.txt")
    parser.add_argument("--force-ascii", action="store_true", help="Escape non-ASCII characters in JSON lines output")
    args = parser.parse_args()
    if args.output_jsonl is None:
        stem = args.transcripts_jsonl.stem + f"_stratified_sample_{args.sample}"
        args.output_jsonl = args.transcripts_jsonl.with_stem(stem)
    return args


def main(args):
    if args.output_jsonl.exists():
        raise FileExistsError(f"File already present at {args.output_jsonl}")
    # transcripts df contains an "ID" column of the form 4800_10003_000000 which is {speaker}_{book}_{audio}
    transcripts = pd.read_json(args.transcripts_jsonl, lines=True, orient="records", dtype={"ID": str})
    # Create a new speaker column by extracting the number before the first underscore from the "ID" column
    transcripts["speaker"] = transcripts["ID"].str.split("_").str[0]
    train, _ = train_test_split(transcripts, train_size=args.sample, stratify=transcripts["speaker"], random_state=SEED)
    train: pd.DataFrame
    train.drop(columns="speaker", inplace=True)
    with open(args.output_jsonl, "x") as f:
        f.write(train.to_json(orient="records", lines=True, force_ascii=args.force_ascii))
    logger.info(f"Wrote {len(train)} lines to {args.output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
