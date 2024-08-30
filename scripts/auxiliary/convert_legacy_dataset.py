#!/usr/bin/env python

import json
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import Any

from tqdm import tqdm

from sardalign.constants import (
    ALIGNMENT_END_TIME_KEY,
    ALIGNMENT_KEY,
    ALIGNMENT_START_TIME_KEY,
    NORMALIZED_KEY,
    SPEECH_TOKENS_KEY,
    TOKENIZED_KEY,
    UROMAN_KEY,
)


def process_line(indexed_line: tuple[int, str]) -> tuple[int, str]:
    # use of index allows order preservation with parallelisation
    index, line = indexed_line
    sample: dict[str, Any] = json.loads(line.strip())
    alignments: list[tuple[str, tuple[float, float]]] = sample.pop(ALIGNMENT_KEY)
    tokenized: list[str] = []
    alignment_starts: list[float] = []
    alignment_ends: list[float] = []
    for token, (alignment_start, alignment_end) in alignments:
        tokenized.append(token)
        alignment_starts.append(alignment_start)
        alignment_ends.append(alignment_end)
    sample[TOKENIZED_KEY] = tokenized
    # order esp. for display in HF datasets
    sample[NORMALIZED_KEY] = sample.pop(NORMALIZED_KEY)
    sample[UROMAN_KEY] = sample.pop(UROMAN_KEY)
    sample[SPEECH_TOKENS_KEY] = sample.pop(SPEECH_TOKENS_KEY)
    sample[ALIGNMENT_START_TIME_KEY] = alignment_starts
    sample[ALIGNMENT_END_TIME_KEY] = alignment_ends
    return (index, json.dumps(sample))


def process_chunk(chunk: list[tuple[int, str]]) -> list[tuple[int, str]]:
    return [process_line(indexed_line) for indexed_line in tqdm(chunk, desc="Processing chunk")]


def main(input_file: Path, output_file: Path) -> None:
    if output_file.exists():
        raise FileExistsError(f"Existing output JSON lines file at {output_file!s}")
    num_cores: int = cpu_count()
    with open(input_file, "r") as f:
        indexed_lines: list[tuple[int, str]] = list(enumerate(tqdm(f, desc=f"Reading in {input_file!s}")))
    chunk_size: int = max(1, len(indexed_lines) // num_cores)
    chunks: list[list[tuple[int, str]]] = [
        indexed_lines[i : i + chunk_size] for i in range(0, len(indexed_lines), chunk_size)
    ]
    with Pool(num_cores) as pool:
        results: list[list[tuple[int, str]]] = pool.map(process_chunk, chunks)
    # Flatten results and sort by index
    all_results: list[tuple[int, str]] = [item for sublist in results for item in sublist]
    all_results.sort(key=lambda x: x[0])
    # Write sorted results to output file
    with open(output_file, "x") as f:
        for _, line in tqdm(all_results, desc=f"Writing to {output_file!s}"):
            f.write(line + "\n")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Convert legacy JSON lines manifest to new format")
    parser.add_argument("input_file", type=Path, help="Path to the legacy JSON lines manifest file")
    parser.add_argument(
        "--output_file", type=Path, default=None, help="Path to output the converted JSON lines manifest file"
    )
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.input_file.with_stem(args.input_file.stem + "_converted")
    return args


if __name__ == "__main__":

    args = parse_args()
    main(args.input_file, args.output_file)
