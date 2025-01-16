#!/usr/bin/env python

"""
{
  "audio_id": "61-70968-0000",
  "text": "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT",
  "speaker_id": "61",
  "chapter_id": 70968,
  "path": "/mnt/scratch-artemis/anilkeshwani/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac"
}
"""

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import count_lines


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def generate_librispeech_jsonl(ls_split_dir: Path, output_jsonl: Path):
    """Generate JSON manifest from an extracted LibriSpeech split e.g. test-clean."""
    trans_txts = ls_split_dir.rglob(r"*.trans.txt")
    f_jsonl = output_jsonl.open("x")
    n_samples = 0
    for trans_txt in trans_txts:
        with open(trans_txt, "r") as f_trans:
            for line in f_trans:
                if line:
                    line = line.strip()
                    audio_id, transcript = line.split(" ", 1)  # maximum of 1 split
                    speaker_id, chapter_id, audio_idx = [int(el) for el in audio_id.split("-")]
                    audio_file = f"{audio_id}.flac"
                    audio_path = str(ls_split_dir / f"{speaker_id}/{chapter_id}/{audio_file}")
                    json.dump(
                        {
                            "id": audio_id,
                            "language": "en",
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "file": audio_file,
                            "path": audio_path,
                            "text": transcript,
                        },
                        f_jsonl,
                        ensure_ascii=False,
                        sort_keys=False,
                    )
                    f_jsonl.write("\n")
                    n_samples += 1
    f_jsonl.close()
    n_written = count_lines(output_jsonl)
    if n_written != n_samples:
        raise RuntimeError(f"Expected to write {n_samples} samples but wrote {n_written} samples")
    LOGGER.info(f"Wrote {n_samples} JSON lines entries to {output_jsonl}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("ls_split_dir", type=Path, help="Path to the LibriSpeech split directory")
    parser.add_argument("--output_jsonl", type=Path, required=True, help="Path to the output JSON lines file")
    return parser.parse_args()


if __name__ == "__main__":
    generate_librispeech_jsonl(**vars(parse_args()))
