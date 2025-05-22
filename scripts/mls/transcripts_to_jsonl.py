#!/usr/bin/env python

import logging
import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any

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
    parser.add_argument("--chunksize", type=int, default=100, help="Chunk size for parallel processing.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if not args.audio_dir.is_absolute():
        raise ValueError("audio-dir must be an absolute path")

    if args.output_jsonl is None:
        head_suffix = f"_head_{args.head}" if args.head != -1 else ""
        stem = args.transcripts.stem + head_suffix
        args.output_jsonl = args.transcripts.with_stem(stem).with_suffix(".jsonl")
    return args


def _read_transcripts_file(args: Namespace) -> list[str]:
    transcript_lines: list[str] = args.transcripts.read_text().splitlines()
    if args.head > 0:
        transcript_lines = transcript_lines[: args.head]
    if len(transcript_lines) == 0:
        raise ValueError(f"Transcripts file {args.transcripts} is empty")
    if args.verbose:
        LOGGER.info(f"Read {len(transcript_lines)} lines from {args.transcripts}")
    return transcript_lines


def _process_transcript_line(line: str, field_delimiter: str, audio_dir: Path, audio_ext: str) -> dict | None:
    """Processes a single transcript line into a dictionary."""
    try:
        mls_id, transcript = line.strip().split(field_delimiter)
        audio_path = mls_id_to_path(mls_id, audio_dir, suffix=audio_ext)
        sample_dict: dict[str, Any] = {
            "ID": mls_id,
            "transcript": transcript,
        }
        try:
            audio_file_info = sox.file_info.info(audio_path)
            sample_dict.update(
                {
                    "duration": audio_file_info.get("duration"),
                    "sample_rate": audio_file_info.get("sample_rate"),
                    "num_samples": audio_file_info.get("num_samples"),
                }
            )
        except OSError:
            LOGGER.warning(f"Skipped metadata for missing audio {audio_path}")
            sample_dict.update({"duration": None, "sample_rate": None, "num_samples": None})
        return sample_dict
    except Exception as e:
        LOGGER.exception(f"Failed to process line: {line}. Error: {e}")
        return None


def main(args: Namespace):
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")
    transcript_lines = _read_transcripts_file(args)
    _fn = partial(
        _process_transcript_line,
        field_delimiter=args.field_delimiter,
        audio_dir=args.audio_dir,
        audio_ext=args.audio_ext,
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(_fn, transcript_lines, chunksize=args.chunksize),
                total=len(transcript_lines),
                desc="Processing transcripts in parallel",
            )
        )

    mls = [r for r in results if r is not None]  # filters out any None results from failures; catch these via logging

    if len(mls) != len(transcript_lines):
        LOGGER.warning(f"Processed {len(mls):,} samples out of {len(transcript_lines):,} lines in the transcripts file")
    else:
        LOGGER.info(f"Processed {len(mls):,} samples from the transcripts file")

    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output at {args.output_jsonl.parent}")

    write_jsonl(args.output_jsonl, mls)
    LOGGER.info(f"Wrote {len(mls)} samples to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
