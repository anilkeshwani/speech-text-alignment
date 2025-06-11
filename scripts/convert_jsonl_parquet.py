#!/usr/bin/env python

import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd


def convert_jsonl_to_parquet(jsonl_path: Path, output_path: Path):
    try:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Load and save
        df = pd.read_json(jsonl_path, dtype=False, lines=True)
        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        print(f"✓ Converted: {jsonl_path} -> {output_path}")
    except Exception as e:
        print(f"✗ Failed: {jsonl_path} ({e})")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Convert a JSON Lines file to Apache Parquet.")
    parser.add_argument("jsonl_path", type=Path, help="Path to the JSONL file")
    parser.add_argument("output_path", type=Path, help="Path to the output Parquet file")
    return parser.parse_args()


if __name__ == "__main__":
    convert_jsonl_to_parquet(**vars(parse_args()))
