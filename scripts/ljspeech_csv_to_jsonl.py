#!/usr/bin/env python

import csv
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Path to the LJSpeech metadata CSV file")
    parser.add_argument("--jsonl", required=True, type=Path, help="Path to output the JSON Lines file")
    return parser.parse_args()


def write_jsonl(jsonl: Path, samples: list[dict], mode="x", encoding="utf-8", ensure_ascii=False):
    with open(jsonl, mode=mode, encoding=encoding) as f:
        f.write("\n".join(json.dumps(sd, ensure_ascii=ensure_ascii) for sd in samples) + "\n")


def main(args: Namespace) -> None:
    LJSPEECH_CSV_DELIMITER = "|"
    LJSPEECH_QUOTECHAR = None
    dataset: list[dict] = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=LJSPEECH_CSV_DELIMITER, quotechar=LJSPEECH_QUOTECHAR)
        for i, line in enumerate(reader):
            ID, transcript, normalized_transcription = line
            dataset.append({"ID": ID, "transcript": transcript, "normalized_transcription": normalized_transcription})
    write_jsonl(args.jsonl, dataset)


if __name__ == "__main__":
    main(parse_args())
