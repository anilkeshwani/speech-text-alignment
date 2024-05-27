#!/usr/bin/env python

import logging
import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import PROJECT_ROOT
from sardalign.text_normalization import text_normalize
from sardalign.utils import read_jsonl, write_jsonl
from sardalign.utils.uroman import post_process_uroman, RomFormat, Uroman
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
    parser.add_argument("input_jsonl", type=Path, help="Path to input JSON lines file")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Output JSON lines path")
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("--n-processes", type=int, default=None, help="Number of parallel processes")
    parser.add_argument("--text-key", type=str, default="transcript", help="Key of text field in JSON lines manifest")
    parser.add_argument("--uroman-key", type=str, default="uroman", help="Key for uroman tokens in JSON lines manifest")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=None,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    args = parser.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = args.input_jsonl.with_stem(args.input_jsonl.stem + "_uroman")
    if args.n_processes is None:
        args.n_processes = max(1, mp.cpu_count() - 1)
    return args


UROMAN_DATA_DIR = PROJECT_ROOT / "submodules" / "uroman" / "data"


def uromanize_chunk(chunk: list[list[str]], lang: str, uroman=Uroman(UROMAN_DATA_DIR)):
    chunk_results = []
    for nt in chunk:
        chunk_results.append(
            post_process_uroman(
                [
                    uroman.romanize_string_core(token, lcode=lang, rom_format=RomFormat.STR, cache_p=False)
                    for token in nt
                ],
                normalize_uroman_post=True,
            )
        )
    return chunk_results


def normalize_chunk(chunk: list[list[str]], lang: str):
    chunk_results: list[list[str]] = []
    for transcripts in chunk:
        chunk_results.append([text_normalize(token, lang) for token in transcripts])
    return chunk_results


def main(args):
    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output at {args.output_jsonl.parent}")
    dataset = read_jsonl(args.input_jsonl)
    LOGGER.info(f"Read {len(dataset)} lines from {args.input_jsonl}")

    transcripts_s: list[list[str]] = []
    for s in tqdm(dataset, desc="Tokenizing dataset"):
        transcripts_s.append(s[args.text_key].strip().split(args.token_delimiter))

    chunk_size = len(transcripts_s) // args.n_processes

    transcripts_s_chunks = [transcripts_s[i : i + chunk_size] for i in range(0, len(transcripts_s), chunk_size)]
    with mp.Pool(processes=args.n_processes) as pool:
        results = list(
            tqdm(
                pool.starmap(normalize_chunk, [(chunk, args.lang) for chunk in transcripts_s_chunks]),
                total=args.n_processes,
                desc="Normalizing transcripts",
            )
        )

    with mp.Pool(processes=args.n_processes) as pool:
        results = list(
            tqdm(
                pool.starmap(uromanize_chunk, [(chunk, args.lang) for chunk in results]),
                total=args.n_processes,
                desc="Processing uroman",
            )
        )

    tokens_s = [item for sublist in results for item in sublist]

    dataset = [s | {args.uroman_key: tokens} for s, tokens in zip(dataset, tokens_s)]

    write_jsonl(args.output_jsonl, dataset)
    LOGGER.info(f"Wrote uromanized filelist to {args.output_jsonl}")


if __name__ == "__main__":
    main(parse_args())
