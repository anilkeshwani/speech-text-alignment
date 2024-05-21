#!/usr/bin/env python

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from pprint import pprint


def count_file_types(archive_file_list: Path):
    files = archive_file_list.read_text().splitlines()
    file_type_counts = Counter([p.split(".")[-1] for p in files])
    pprint(dict(file_type_counts))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("archive_file_list", type=Path, help="Path to achive file list - files only, no directories")
    args = parser.parse_args()
    count_file_types(args.archive_file_list)
