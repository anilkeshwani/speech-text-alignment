#!/usr/bin/env python

"""
Script to
1. validate Python uromanization implementation from Ulf Hermjakob; and
2. benchmark original Perl vs Python implementations
"""

import re
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sardalign.text_normalization import text_normalize
from sardalign.utils import read_jsonl, write_jsonl
from sardalign.utils.align import get_uroman_tokens, normalize_uroman
from sardalign.utils.uroman import RomFormat, Uroman
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to input JSON lines file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("--text-key", type=str, default="transcript", help="Key of text field in JSON lines manifest")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=None,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    parser.add_argument("--uroman-path", type=Path, default=None, help="Location to uroman/bin")
    parser.add_argument("--uroman-data-dir", type=Path, default=None, help="Location to uroman/bin")
    parser.add_argument("--sample", type=int, default=None, help="Use a sample of the dataset for testing purposes")
    args = parser.parse_args()
    if args.uroman_path is None:
        args.uroman_path = Path(__file__).parents[1] / "submodules" / "uroman" / "bin"
    return args


def post_process_uroman(tokens: list[str], normalize_uroman_post: bool):
    tokens_pp = [re.sub(r"\s+", " ", " ".join(s.strip())).strip() for s in tokens]
    if normalize_uroman_post:
        tokens_pp = [normalize_uroman(s) for s in tokens_pp]
    return tokens_pp


def main(args):
    lap = time.perf_counter()
    print(f"Script began at: {lap:.2f}")
    dataset = read_jsonl(args.jsonl)
    print(f"Took {time.perf_counter() - lap:.2f}s - Reading JSON lines")
    lap = time.perf_counter()
    if args.sample is not None:
        dataset = dataset[: args.sample]
        print(f"Took {time.perf_counter() - lap:.2f}s - Taking head of dataset")
        lap = time.perf_counter()
    print(f"Read {len(dataset)} lines from {args.jsonl}")
    transcripts_s: list[list[str]] = []
    for s in tqdm(dataset, desc="Tokenizing dataset"):
        transcripts_s.append(s[args.text_key].strip().split(args.token_delimiter))
    print(f"Took {time.perf_counter() - lap:.2f}s - Tokenising dataset")
    lap = time.perf_counter()
    norm_transcripts_s: list[list[str]] = []
    for transcripts in tqdm(transcripts_s, desc="Normalizing transcripts"):
        norm_transcripts_s.append([text_normalize(token, args.lang) for token in transcripts])
    print(f"Took {time.perf_counter() - lap:.2f}s - Normalizing transcripts")
    lap = time.perf_counter()

    # # Implementation via Perl script w/ a lot of IO
    # tokens_s_perl = []
    # for nt in tqdm(norm_transcripts_s, desc="Getting uroman tokens for transcripts"):
    #     tokens_s_perl.append(get_uroman_tokens(nt, args.uroman_path, args.lang))
    # print(
    #     f"Took {time.perf_counter() - lap:.2f}s - "
    #     "Romanizing transcripts: Implementation via Perl script w/ a lot of IO"
    # )
    lap = time.perf_counter()

    # # Implementation via Python w/o IO
    uroman = Uroman(args.uroman_data_dir)
    tokens_s_python = []
    for nt in tqdm(norm_transcripts_s, desc="Getting uroman tokens for transcripts"):
        tokens_s_python.append(
            post_process_uroman(
                [uroman.romanize_string_core(_, lcode=args.lang, rom_format=RomFormat.STR, cache_p=False) for _ in nt],
                normalize_uroman_post=True,
            )
        )
    print(f"Took {time.perf_counter() - lap:.2f}s - " "Romanizing transcripts: Implementation in native Python w/o IO")
    lap = time.perf_counter()

    # # Write to disk
    # dataset_perl = [s | {"uroman_tokens": tokens_perl} for s, tokens_perl in zip(dataset, tokens_s_perl)]
    # args.out_dir.mkdir(parents=True, exist_ok=True)
    # write_jsonl(args.out_dir / "uromanized_perl.jsonl", dataset_perl)

    dataset_python = [s | {"uroman_tokens": tokens_python} for s, tokens_python in zip(dataset, tokens_s_python)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "uromanized_python.jsonl", dataset_python)

    print(f"Script ended at: {lap:.2f}")


if __name__ == "__main__":
    main(parse_args())
