#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import count_lines
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
    # e.g. /mnt/scratch-artemis/anilkeshwani/data/GigaSpeech_HF/GigaSpeech.jsonl
    parser.add_argument("jsonl", type=Path, help="Path to the JSON lines file")
    parser.add_argument("--head", type=int, default=None, help="Check only head samples")
    parser.add_argument("--check-duration", action="store_true", help="Check audio duration via SoX")
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=0.01,
        help="Tolerance for difference when checking audio durations",
    )
    args = parser.parse_args()
    return args


def main(jsonl: Path, check_duration: bool, head: int, duration_tolerance: float | None = None):
    n = count_lines(jsonl)
    missing = 0
    if check_duration:
        assert duration_tolerance is not None, "Must specify tolerance for duration diffs when checking audio durations"
    with open(jsonl) as f:
        for i, line in tqdm(enumerate(f), total=head if head is not None else n):
            sample = json.loads(line)
            if head is not None and i >= head:
                break
            audio_exists = Path(sample["path"]).exists()
            if not audio_exists:
                LOGGER.warning(f"Audio missing: {sample['path']}")
                missing += 1
            if check_duration and audio_exists:
                dur: float = sox.file_info.duration(sample["path"])
                dur_metadata: float = sample["end_time"] - sample["begin_time"]
                if (dur - dur_metadata) > duration_tolerance:
                    LOGGER.warning(f"Duration does not match for {sample['path']}: {dur:.f2} vs {dur_metadata:.f2}")
                else:
                    LOGGER.info(f"Durations match for {sample['path']}")


if __name__ == "__main__":
    main(**vars(parse_args()))
