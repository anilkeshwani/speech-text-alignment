#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import pandas as pd


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("transcripts_jsonl", type=Path, help="Transcripts JSON lines file")
    args = parser.parse_args()
    return args


def main(args):
    transcripts = pd.read_json(args.transcripts_jsonl, lines=True, orient="records", dtype={"ID": str})
    if "speaker" not in transcripts.columns:
        transcripts["speaker"] = transcripts["ID"].str.split("_").str[0].astype(int)
    pprint(transcripts.speaker.value_counts().to_dict(), sort_dicts=False)


if __name__ == "__main__":
    main(parse_args())
