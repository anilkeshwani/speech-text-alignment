#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import (
    ALIGNMENT_KEY,
    HUBERT_DOWNSAMPLING_RATIO,
    HUBERT_TOKEN_FSTRING,
    SAMPLING_FREQ,
    SPEECH_TOKENS_KEY,
)
from sardalign.utils import count_lines, read_jsonl
from sardalign.utils.align import span_times_to_hubert_idxs
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
    parser.add_argument("input_jsonl", type=Path, help="Path to jsonl with text alignments and HuBERT speech tokens")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Path to write interleaved data")
    args = parser.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = args.input_jsonl.with_stem(args.input_jsonl.stem + "_interleaved")
    return args


def main(args: Namespace):
    dataset = read_jsonl(args.input_jsonl)
    interleaved_dataset: list[str] = []
    for sample in tqdm(dataset, desc="Generating interleaved text-speech samples"):
        interleaved_tokens = []
        speech_tokens = sample[SPEECH_TOKENS_KEY]
        alignment: dict[str, tuple[int, int]] = sample[ALIGNMENT_KEY]
        for i, (token, (sp_start, sp_end)) in enumerate(alignment.items()):
            start_idx, end_idx = span_times_to_hubert_idxs((sp_start, sp_end), SAMPLING_FREQ, HUBERT_DOWNSAMPLING_RATIO)
            if i // 5 % 2 == 0:
                interleaved_tokens += [HUBERT_TOKEN_FSTRING.format(ht) for ht in speech_tokens[start_idx:end_idx]]
            else:
                interleaved_tokens.append(token)
        interleaved_segment = " ".join(interleaved_tokens)

        from pathlib import Path

        import sox
        from sardalign.utils import mls_id_to_path

        for _ in interleaved_tokens:
            print(_)
        print(sample["transcript"])
        print(
            sox.file_info.duration(
                mls_id_to_path(sample["ID"], Path("/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/audio"))
            )
        )
        breakpoint()

        interleaved_dataset.append(interleaved_segment)
    LOGGER.info(f"Wrote {count_lines(args.output_jsonl)} lines to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
