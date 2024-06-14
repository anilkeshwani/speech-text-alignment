#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from itertools import zip_longest
from pathlib import Path

import numpy as np
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import (
    ALIGNMENT_KEY,
    HUBERT_DOWNSAMPLING_RATIO,
    HUBERT_TOKEN_FSTRING,
    MEGATRON_TEXT_KEY,
    MODALITY_TOKEN_SPEECH,
    MODALITY_TOKEN_TEXT,
    SAMPLING_FREQ,
    SEED,
    SPEECH_TOKENS_KEY,
    TEXT_KEY,
    TOKEN_DELIMITER,
)
from sardalign.utils import count_lines, read_jsonl, write_jsonl
from sardalign.utils.align import span_times_to_hubert_idxs
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

MEAN_MLS_SEQ_LEN: float = 39.43  # mean sequence length (in tokens) of the MLS stratified sample; 25% of en trainset
BINOM_PROB: float = 0.1  # fraction of sequence to make up subspans


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input_jsonl", type=Path, help="Path to jsonl with text alignments and HuBERT speech tokens")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Path to write interleaved data")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=TOKEN_DELIMITER,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    parser.add_argument(
        "--no-modality-tokens",
        action="store_false",
        dest="use_modality_tokens",
        help="Do no prepend special modality tokens to spans of text/speech tokens",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()
    return args


def get_span_idxs_binomial(n: int, p: float, seq_len: int, seed: int = SEED) -> list[int]:
    prng = np.random.default_rng(seed)
    subspan_idxs = np.maximum(prng.binomial(int(n), p, size=seq_len), 1).cumsum()
    return [0] + subspan_idxs[subspan_idxs < (seq_len - 1)].tolist() + [(seq_len - 1)]


def interleave_dataset(
    input_jsonl: Path,
    output_jsonl: Path | None = None,
    use_modality_tokens: bool = True,
    token_delimiter: str | None = TOKEN_DELIMITER,
    seed: int = SEED,
) -> Path:
    if output_jsonl is None:
        output_jsonl = input_jsonl.with_stem(input_jsonl.stem + "_interleaved")

    dataset = read_jsonl(input_jsonl)
    interleaved_dataset: list[dict] = []

    for i, sample in enumerate(tqdm(dataset, desc="Generating interleaved text-speech samples")):
        start_with_text = i % 2 == 0  # even-indexed samples start w/ text
        speech_tokens = sample[SPEECH_TOKENS_KEY]
        tokens = sample[TEXT_KEY].split(token_delimiter)
        alignment = list(sample[ALIGNMENT_KEY].values())
        if len(tokens) != len(alignment):
            raise ValueError(
                f"Tokens and alignment lengths differ: {len(tokens)} vs. {len(alignment)} for sample "
                f"{input_jsonl!s}#{i + 1}"
            )
        span_idxs = get_span_idxs_binomial(int(MEAN_MLS_SEQ_LEN), BINOM_PROB, len(tokens), seed)
        idxs1, idxs2 = zip(span_idxs[:-1:2], span_idxs[1::2]), zip(span_idxs[1:-1:2], span_idxs[2::2])  # BUG truncates
        print(f"{len(tokens)}")
        text_idxs, hubert_idxs = (idxs1, idxs2) if start_with_text else (idxs2, idxs1)
        text_spans: list[str] = [" ".join(tokens[start_idx:end_idx]) for start_idx, end_idx in text_idxs]
        hubert_spans: list[str] = []
        for start_idx, end_idx in hubert_idxs:
            (start_time, _), (_, end_time) = alignment[start_idx], alignment[end_idx]
            start_idx_hu, end_idx_hu = span_times_to_hubert_idxs(
                (start_time, end_time), SAMPLING_FREQ, HUBERT_DOWNSAMPLING_RATIO
            )
            speech_tokens_span = speech_tokens[start_idx_hu:end_idx_hu]
            hubert_spans.append("".join([HUBERT_TOKEN_FSTRING.format(ht) for ht in speech_tokens_span]))

        if use_modality_tokens:
            text_spans = [" ".join((MODALITY_TOKEN_TEXT, text_span)) for text_span in text_spans]
            hubert_spans = [" ".join((MODALITY_TOKEN_SPEECH, hubert_span)) for hubert_span in hubert_spans]

        mm_spans = (text_spans, hubert_spans) if start_with_text else (hubert_spans, text_spans)
        interleaved_segment = " ".join([span for spans in zip_longest(*mm_spans, fillvalue="") for span in spans])
        interleaved_dataset.append({MEGATRON_TEXT_KEY: interleaved_segment})

    write_jsonl(output_jsonl, interleaved_dataset, "w")
    LOGGER.info(f"Wrote {count_lines(output_jsonl)} lines to {output_jsonl!s}")
    return output_jsonl


if __name__ == "__main__":
    args = parse_args()
    interleave_dataset(**vars(args))
