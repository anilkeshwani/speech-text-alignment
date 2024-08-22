#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
from tqdm import tqdm

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import mls_id_to_path, write_jsonl


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True, help="Path to the MLS transcripts.txt file.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing the audio files.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Path to the output JSON lines file.")
    parser.add_argument("--audio-ext", type=str, default=".flac", help="File extension of the audio files.")
    parser.add_argument("--field-delimiter", type=str, default="\t", help="Field delimiter used in transcripts file.")
    parser.add_argument("--head", type=int, default=-1, help="Head samples to take; -1 for all")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # argument validation
    if not args.audio_dir.is_absolute():
        raise ValueError("audio-dir must be an absolute path")  # require absolute paths for now

    # argument resolution
    if args.output_jsonl is None:
        head_suffix = f"_head_{args.head}" if args.head != -1 else ""
        stem = args.transcripts.stem + head_suffix
        args.output_jsonl = args.transcripts.with_stem(stem).with_suffix(".jsonl")
    return args


def main(args: Namespace):
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")
    transcript_lines: list[str] = args.transcripts.read_text().splitlines()
    if args.head > 0:
        transcript_lines = transcript_lines[: args.head]
    mls: list[dict] = []
    for line in tqdm(transcript_lines):
        mls_id, transcript = line.strip().split(args.field_delimiter)
        audio_path = mls_id_to_path(mls_id, args.audio_dir, suffix=args.audio_ext)
        sample_dict = {
            "ID": mls_id,
            "transcript": transcript,
        }
        try:
            audio_file_info = sox.file_info.info(audio_path)
            sample_dict = sample_dict | {
                "duration": audio_file_info.get("duration"),
                "sample_rate": audio_file_info.get("sample_rate"),
                "num_samples": audio_file_info.get("num_samples"),
            }
        except OSError:  # sox raises an OSError and not a FileNotFoundError; think this is a sys call to soxi
            if args.verbose:
                LOGGER.info(f"Skipped addition of metadata for missing audio {audio_path}")
        mls.append(sample_dict)
    write_jsonl(args.output_jsonl, mls)
    LOGGER.info(f"Wrote {len(mls)} samples to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
