#!/usr/bin/env python

import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sardalign.utils import write_jsonl


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Path to the LJSpeech metadata CSV file")
    parser.add_argument("--jsonl", required=True, type=Path, help="Path to output the JSON Lines file")
    parser.add_argument("--add-lang-code", action="store_true", help="Flag: Adds 'en-US' language code as 'lang' field")
    return parser.parse_args()


def main(args: Namespace) -> None:
    LJSPEECH_CSV_DELIMITER = "|"
    LJSPEECH_QUOTECHAR = None
    dataset: list[dict] = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=LJSPEECH_CSV_DELIMITER, quotechar=LJSPEECH_QUOTECHAR)
        for i, line in enumerate(reader):
            ID, transcript, normalized_transcription = line
            dataset.append({"ID": ID, "transcript": transcript, "normalized_transcription": normalized_transcription})
    if args.add_lang_code:
        dataset = [s | {"lang": "en-US"} for s in dataset]
    write_jsonl(args.jsonl, dataset)


if __name__ == "__main__":
    main(parse_args())
