#!/usr/bin/env python

import json
import logging
import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import NORMALIZED_KEY, PROJECT_ROOT, TOKEN_DELIMITER_DEFAULT, TOKENIZED_KEY, UROMAN_KEY
from sardalign.text_normalization import text_normalize
from sardalign.utils import read_jsonl, write_jsonl
from sardalign.utils.uroman import post_process_uroman, RomFormat, Uroman


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

UROMAN_DATA_DIR = PROJECT_ROOT / "submodules" / "uroman" / "data"
UROMAN = Uroman(UROMAN_DATA_DIR)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input_jsonl", type=Path, help="Path to input JSON lines file")
    parser.add_argument("--text-key", type=str, required=True, help="Text field key in *input* JSON lines file.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Output JSON lines path")
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("--n-processes", type=int, default=None, help="Number of parallel processes")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=TOKEN_DELIMITER_DEFAULT,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    args = parser.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = args.input_jsonl.with_stem(args.input_jsonl.stem + "_uroman")
    if args.n_processes is None:
        args.n_processes = max(1, mp.cpu_count() - 1)
    return args


def uromanize_chunk(chunk: list[list[str]], lang: str, uroman=UROMAN):
    chunk_results = []
    for nt in tqdm(chunk, desc="Uromanization"):
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
    for transcripts in tqdm(chunk, "Normalization"):
        chunk_results.append([text_normalize(token, lang) for token in transcripts])
    return chunk_results


def main(args):
    if args.output_jsonl.exists():
        raise FileExistsError(f"Uromanized JSON lines output file already exists at {args.output_jsonl}")

    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output at {args.output_jsonl.parent}")

    with open(args.input_jsonl) as f:
        first_sample = json.loads(f.readline())
    if any(new_key in first_sample.keys() for new_key in [NORMALIZED_KEY, TOKENIZED_KEY, UROMAN_KEY]):
        raise RuntimeError(
            "Conflict in key names. "
            f"One of {[NORMALIZED_KEY, TOKENIZED_KEY, UROMAN_KEY]} already present in input data."
        )

    dataset = read_jsonl(args.input_jsonl)
    LOGGER.info(f"Read {len(dataset)} lines from {args.input_jsonl}")

    tokens_s: list[list[str]] = []
    for s in tqdm(dataset, desc="Tokenizing dataset"):
        tokens_s.append(s[args.text_key].strip().split(args.token_delimiter))

    chunk_size = len(tokens_s) // args.n_processes

    tokens_s_chunks = [tokens_s[i : i + chunk_size] for i in range(0, len(tokens_s), chunk_size)]
    with mp.Pool(processes=args.n_processes) as pool:
        normalized_chunks = list(pool.starmap(normalize_chunk, [(chunk, args.lang) for chunk in tokens_s_chunks]))

    normalized_tokens_s = [item for sublist in normalized_chunks for item in sublist]

    with mp.Pool(processes=args.n_processes) as pool:
        results = list(pool.starmap(uromanize_chunk, [(chunk, args.lang) for chunk in normalized_chunks]))

    uroman_tokens_s = [item for sublist in results for item in sublist]

    uromanized_dataset = []
    for s, tokens, normalized_tokens, uroman_tokens in zip(dataset, tokens_s, normalized_tokens_s, uroman_tokens_s):
        if not (len(tokens) == len(normalized_tokens) == len(uroman_tokens)):
            raise RuntimeError("Mismatch in number of tokens, normalized tokens and uromanized tokens.")
        uromanized_dataset.append(
            s | {TOKENIZED_KEY: tokens, NORMALIZED_KEY: normalized_tokens, UROMAN_KEY: uroman_tokens}
        )

    write_jsonl(args.output_jsonl, uromanized_dataset)

    LOGGER.info(f"Wrote uromanized filelist to {args.output_jsonl}")


if __name__ == "__main__":
    main(parse_args())
