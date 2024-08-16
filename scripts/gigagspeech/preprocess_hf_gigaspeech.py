#!/usr/bin/env python

import json
import logging
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import librosa
import nltk
from datasets import load_dataset
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import SAMPLING_FREQ
from sardalign.constants.gigaspeech import GS_GARBAGE_TAGS, GS_HF_DATASET_REPO, GS_PUNCTUATION_MAP
from sardalign.utils import count_lines
from tqdm import tqdm
from truecase import TrueCaser


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
    parser.add_argument("--subset", type=str, required=True, help="Data subset", choices=["xs", "s", "m", "l", "xl"])
    parser.add_argument("--split", type=str, default="train", help="Dataset split")  # TODO Choices?
    parser.add_argument("--keep-garbage", action="store_true", help="Retain segments containing garbage utterances")
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


# NOTE Data instances are structured as follows per the docs: https://huggingface.co/datasets/speechcolab/gigaspeech
# {
#     "segment_id": "YOU0000000315_S0000660",
#     "speaker": "N/A",
#     "text": "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>",
#     "audio": {
#         # in streaming mode 'path' will be 'xs_chunks_0000/YOU0000000315_S0000660.wav'
#         "path": "/home/user/.cache/huggingface/datasets/downloads/extracted/9d48cf31/xs_chunks_0000/YOU0000000315_S0000660.wav",
#         "array": array([0.0005188, 0.00085449, 0.00012207, ..., 0.00125122, 0.00076294, 0.00036621], dtype=float32),
#         "sampling_rate": 16000,
#     },
#     "begin_time": 2941.889892578125,
#     "end_time": 2945.070068359375,
#     "audio_id": "YOU0000000315",
#     "title": "Return to Vasselheim | Critical Role: VOX MACHINA | Episode 43",
#     "url": "https://www.youtube.com/watch?v=zr2n1fLVasU",
#     "source": 2,
#     "category": 24,
#     "original_full_path": "audio/youtube/P0004/YOU0000000315.opus",
# }


# Function to transform GigaSpeech punctuation ["<COMMA>", "<PERIOD>", "<QUESTIONMARK>", "<EXCLAMATIONPOINT>"] to
# " , . ? ! " including removal of the leading space
# Example:
# "text": "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>",
# -> "AS THEY'RE LEAVING, CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY?",


def sub_gs_punct(text: str) -> str:
    """Substitute GigaSpeech punctuation tags to true punctuation."

    Args:
        text (str): Raw input text from GigaSpeech metadata file.

    Returns:
        str: Text with punctuation tags replaced with ASCII punctuation (otherwise unmodified).

    Examples:
        "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>"
        -> "AS THEY'RE LEAVING, CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY?"

    Note:
        Recall the GigaSpeech punctuation tags are ["<COMMA>", "<PERIOD>", "<QUESTIONMARK>", "<EXCLAMATIONPOINT>"]
    """
    for punctuation_tag, punctuation_mark in GS_PUNCTUATION_MAP.items():
        text = re.sub(rf"\s*{punctuation_tag}", punctuation_mark, text)
    return text


def main(args: Namespace):
    nltk.download("punkt")  # required for truecase
    truecaser = TrueCaser()
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")

    ds = load_dataset(GS_HF_DATASET_REPO, args.subset, trust_remote_code=True, cache_dir=args.cache_dir)
    train = ds[args.split]
    n_train = len(train)  # NOTE Equal to len({sample["audio"]["path"] for sample in tqdm(train, total=n_train)})
    LOGGER.info(f"Loaded GigaSpeech {args.subset.upper()} subset with {n_train:,} training samples from Hugging Face")

    if not args.output_jsonl.parent.exists():
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created directory for output JSON lines at: {args.output_jsonl.parent!s}")

    n_processed = 0
    with open(args.output_jsonl, "x") as f:
        for sample in tqdm(train, total=n_train if args.head == -1 else args.head):
            sample: dict[str, Any]

            if not args.keep_garbage and any(tag in sample["text"] for tag in GS_GARBAGE_TAGS):
                LOGGER.info(f"Skipping segment {sample['segment_id']} due to garbage utterance. Text: {sample['text']}")
                continue

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
                    f"Got SR {sample['audio']['sampling_rate']} for segment "
                    f"{sample['segment_id']} at {sample['audio']['path']}"
                )

            sample_dict = {
                "segment_id": sample["segment_id"],
                "text": sample["text"],
                "text_processed": truecaser.get_true_case(
                    sub_gs_punct(sample["text"]).lower(), out_of_vocabulary_token_option="title"
                ),
                "audio_id": sample["audio_id"],
                "path": sample["audio"]["path"],
                "speaker": sample.get("speaker"),
                "begin_time": sample.get("begin_time"),
                "end_time": sample.get("end_time"),
                "title": sample.get("title"),
                "url": sample.get("url"),
                "source": sample.get("source"),
                "category": sample.get("category"),
                "original_full_path": sample.get("original_full_path"),
            }

            if args.include_array:
                sample_dict |= {"array": sample["audio"]["array"].tolist()}

            f.write(json.dumps(sample_dict) + "\n")
            n_processed += 1

    LOGGER.info(f"Wrote {count_lines(args.output_jsonl)} output JSON lines manifest to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
