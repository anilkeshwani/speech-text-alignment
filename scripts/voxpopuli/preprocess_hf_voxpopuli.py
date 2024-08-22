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
from sardalign.constants.voxpopuli import VOXPOPULI_HF_DATASET_REPO
from sardalign.utils import count_lines


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
    parser.add_argument("--subset", type=str, default="en", help="Language subset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")  # TODO Choices?
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


# NOTE Data instances are structured as follows.
# The HF docs provide an example from the Croatian language subset: https://huggingface.co/datasets/facebook/voxpopuli
# {
#     "audio_id": "20180418-0900-PLENARY-3-en_20180418-08:50:36_17",
#     "language": 0,
#     "audio": {
#         "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/df38b2dc476ca70ce60b9497720ca5813ff0d35423ed2dad36707ed85eb3b028/train_part_0/20180418-0900-PLENARY-3-en_20180418-08:50:36_17.wav",
#         "array": array([-0.00030518, 0.00119019, 0.00506592, ..., -0.00036621, -0.00027466, -0.00018311]),
#         "sampling_rate": 16000,
#     },
#     "raw_text": "If you do not address this problem, the ground is there for "
#     "populist nationalist forces to go on growing all over Europe.",
#     "normalized_text": "if you do not address this problem the ground is there "
#     "for populist nationalist forces to go on growing all over "
#     "europe.",
#     "gender": "female",
#     "speaker_id": "124737",
#     "is_gold_transcript": True,
#     "accent": "None",
# }


def main(args: Namespace):
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")

    ds = load_dataset(VOXPOPULI_HF_DATASET_REPO, args.subset, trust_remote_code=True, cache_dir=args.cache_dir)
    train = ds[args.split]
    n_train = len(train)
    LOGGER.info(f"Loaded VoxPopuli {args.subset} subset with {n_train:,} training samples from Hugging Face")

    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output JSON lines at: {args.output_jsonl.parent!s}")

    n_processed = 0
    with open(args.output_jsonl, "x") as f:
        for sample in tqdm(train, total=n_train if args.head == -1 else args.head):
            sample: dict[str, Any]

            if args.head != -1 and n_processed >= args.head:
                LOGGER.info(
                    f"Processed {args.head:,} / {n_train} total samples as specified by --head. Breaking early."
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
