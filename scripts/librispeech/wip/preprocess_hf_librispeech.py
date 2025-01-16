#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import librosa
from datasets import load_dataset
from tqdm import tqdm

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import SAMPLING_FREQ
from sardalign.utils import count_lines


LIBRISPEECH_HF_DATASET_REPO: str = "openslr/librispeech_asr"

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to the output JSON lines file")
    parser.add_argument("--config", type=str, required=True, help="Language subset")
    parser.add_argument("--split", type=str, required=True, help="Dataset split")  # TODO Choices?
    parser.add_argument("--head", type=int, default=-1, help="Head audio files to preprocess; -1 for all")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument(
        "--include-array", action="store_true", help="Include the Hugging Face array in the JSON lines output"
    )
    parser.add_argument("--check-array", action="store_true", help="Load audio and check HF array and SR entries match")
    args = parser.parse_args()

    if args.head != -1 and args.head <= 0:
        raise ValueError(f"Invalid to specify {args.head} samples for head. Pass a positive integer or -1 (all data)")

    if args.check_array and not args.include_array:
        raise ValueError("Must specify --include-array flag to check the metadata array matches the waveform on disk")

    if args.head > 0 and "head" not in str(args.output_jsonl.name):
        LOGGER.warning(f"--head {args.head} specified but 'head' not in output JSON lines filename")

    return args


"""
# NOTE Data instances are structured as follows.
{'chapter_id': 141231,
 'file': '/home/patrick/.cache/huggingface/datasets/downloads/extracted/b7ded9969e09942ab65313e691e6fc2e12066192ee8527e21d634aca128afbe2/dev_clean/1272/141231/1272-141231-0000.flac',
  'audio': {'path': '/home/patrick/.cache/huggingface/datasets/downloads/extracted/b7ded9969e09942ab65313e691e6fc2e12066192ee8527e21d634aca128afbe2/dev_clean/1272/141231/1272-141231-0000.flac',
  'array': array([-0.00048828, -0.00018311, -0.00137329, ...,  0.00079346,
          0.00091553,  0.00085449], dtype=float32),
  'sampling_rate': 16000},
 'id': '1272-141231-0000',
 'speaker_id': 1272,
 'text': 'A MAN SAID TO THE UNIVERSE SIR I EXIST'}
"""


def main(args: Namespace):
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")

    ds = load_dataset(LIBRISPEECH_HF_DATASET_REPO, args.config, trust_remote_code=True, cache_dir=args.cache_dir)
    split_ls = ds[args.split]
    n_split_ls = len(split_ls)
    LOGGER.info(f"Loaded LibriSpeech {args.config} subset with {n_split_ls:,} samples from Hugging Face")

    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output JSON lines at: {args.output_jsonl.parent!s}")

    n_processed = 0
    with open(args.output_jsonl, "x") as f:
        for sample in tqdm(split_ls, total=n_split_ls if args.head == -1 else args.head):
            sample: dict[str, Any]

            if args.head != -1 and n_processed >= args.head:
                LOGGER.info(
                    f"Processed {args.head:,} / {n_split_ls} total samples as specified by --head. Breaking early."
                )
                break

            if args.check_array:
                wave, sr = librosa.load(sample["audio"]["path"], sr=None)
                if not (wave == sample["audio"]["array"]).all():
                    raise ValueError(
                        f"Hugging Face audio array does not match loaded waveform for {sample['segment_id']}"
                    )

            if sample["audio"]["sampling_rate"] != SAMPLING_FREQ:
                raise ValueError(
                    f"Got SR {sample['audio']['sampling_rate']} for audio "
                    f"{sample['audio_id']} at {sample['audio']['path']}"
                )

            sample_dict = {
                "audio_id": sample["audio_id"],
                "language": sample["language"],
                "raw_text": sample["raw_text"],
                "normalized_text": sample["normalized_text"],
                "gender": sample.get("gender"),
                "speaker_id": sample.get("speaker_id"),
                "is_gold_transcript": sample.get("is_gold_transcript"),
                "accent": sample.get("accent"),
                "path": sample["audio"]["path"],
            }

            if args.include_array:
                sample_dict |= {"array": sample["audio"]["array"].tolist()}

            f.write(json.dumps(sample_dict) + "\n")
            n_processed += 1

    LOGGER.info(f"Wrote {count_lines(args.output_jsonl)} output JSON lines manifest to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
