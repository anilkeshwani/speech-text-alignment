#!/usr/bin/env python

import concurrent.futures
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from functools import partial
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
    parser.add_argument("--text-key", type=str, default="transcript", help="Key of text field in JSON lines manifest")
    parser.add_argument("--uroman-key", type=str, default="uroman", help="Key for uroman tokens in JSON lines manifest")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=None,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    parser.add_argument("--uroman-data-dir", type=Path, default=None, help="Path to uroman/data")
    args = parser.parse_args()
    if args.uroman_data_dir is None:
        args.uroman_data_dir = PROJECT_ROOT / "submodules" / "uroman" / "data"
    if args.output_jsonl is None:
        args.output_jsonl = args.input_jsonl.with_stem(args.input_jsonl.stem + "_uroman")
    return args


UROMAN = Uroman(PROJECT_ROOT / "submodules" / "uroman" / "data")

romanize_string_core_partial = partial(
    UROMAN.romanize_string_core, lcode="eng", rom_format=RomFormat.STR, cache_p=False
)


def _parallel_post_process_uroman(nt: list[str]):
    return post_process_uroman([romanize_string_core_partial(token) for token in nt], normalize_uroman_post=True)


def main(args):
    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output at {args.output_jsonl.parent}")
    dataset = read_jsonl(args.input_jsonl)
    LOGGER.info(f"Read {len(dataset)} lines from {args.input_jsonl}")
    transcripts_s: list[list[str]] = []
    for s in tqdm(dataset, desc="Tokenizing dataset"):
        transcripts_s.append(s[args.text_key].strip().split(args.token_delimiter))
    norm_transcripts_s: list[list[str]] = []
    for transcripts in tqdm(transcripts_s, desc="Normalizing transcripts"):
        norm_transcripts_s.append([text_normalize(token, args.lang) for token in transcripts])

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        tokens_s = []
        for nt in norm_transcripts_s:
            tokens_s.append(list(executor.map(_parallel_post_process_uroman, nt)))

    # Write to disk
    dataset = [s | {args.uroman_key: tokens} for s, tokens in zip(dataset, tokens_s)]

    write_jsonl(args.output_jsonl, dataset)
    LOGGER.info(f"Wrote uromanized filelist to {args.output_jsonl}")


if __name__ == "__main__":
    main(parse_args())
