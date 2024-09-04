#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from itertools import groupby, zip_longest
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import (
    ALIGNMENT_END_TIME_KEY,
    ALIGNMENT_START_TIME_KEY,
    HUBERT_DOWNSAMPLING_RATIO,
    MODALITY_TOKEN_SPEECH,
    MODALITY_TOKEN_TEXT,
    SAMPLING_FREQ,
    SEED,
    SPEECH_TOKENS_KEY,
    TOKENIZED_KEY,
)
from sardalign.constants.megatron import MEGATRON_TEXT_KEY
from sardalign.utils import count_lines, dsu2pua
from sardalign.utils.align import times_to_hubert_idxs


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
        "--no-modality-tokens",
        action="store_false",
        dest="use_modality_tokens",
        help="Do no prepend special modality tokens to spans of text/speech tokens",
    )
    parser.add_argument(
        "--no-deduplication",
        action="store_false",
        dest="deduplicate",
        help="Do not deduplicate consecutive DSUs",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = args.input_jsonl.with_stem(args.input_jsonl.stem + "_interleaved")
    return args


def get_span_idxs_binomial(n: int, p: float, seq_len: int, seed: int = SEED) -> list[int]:
    prng = np.random.default_rng(seed)
    subspan_idxs = np.maximum(prng.binomial(n, p, size=seq_len), 1).cumsum()
    return [0] + subspan_idxs[subspan_idxs < seq_len].tolist() + [seq_len]


def interleave_dataset(
    input_jsonl: Path, output_jsonl: Path, use_modality_tokens: bool, deduplicate: bool, seed: int = SEED
) -> Path:
    with open(input_jsonl, mode="r") as f, open(output_jsonl, mode="x") as g:
        for i, line in enumerate(tqdm(f, desc="Interleaving text-speech samples", total=count_lines(input_jsonl))):
            sample = json.loads(line)
            start_with_text = i % 2 == 0  # even-indexed samples start with text tokens
            tokens = sample[TOKENIZED_KEY]
            align_t_starts = sample[ALIGNMENT_START_TIME_KEY]
            align_t_ends = sample[ALIGNMENT_END_TIME_KEY]
            speech_tokens: list[int] = sample[SPEECH_TOKENS_KEY]
            span_idxs = get_span_idxs_binomial(int(MEAN_MLS_SEQ_LEN), BINOM_PROB, len(tokens), seed)
            # idxs: list of 2-tuples of start and end indices of subspans e.g. [(0, 4), (11, 16), (21, 25), (28, 31)]
            idxs1, idxs2 = zip(span_idxs[:-1:2], span_idxs[1::2]), zip(span_idxs[1:-1:2], span_idxs[2::2])
            text_idxs, hubert_idxs = (idxs1, idxs2) if start_with_text else (idxs2, idxs1)
            text_spans: list[str] = [" ".join(tokens[start_idx:end_idx]) for start_idx, end_idx in text_idxs]
            hubert_spans: list[str] = []
            for start_idx, end_idx in hubert_idxs:
                start_idx_hu, end_idx_hu = times_to_hubert_idxs(
                    (align_t_starts[start_idx], align_t_ends[end_idx - 1]),
                    SAMPLING_FREQ,
                    HUBERT_DOWNSAMPLING_RATIO,
                )
                sp_tkns_spn = speech_tokens[start_idx_hu:end_idx_hu]
                if deduplicate:
                    sp_tkns_spn = [k for k, g in groupby(sp_tkns_spn)]
                hubert_spans.append("".join([dsu2pua(sp_tkn) for sp_tkn in sp_tkns_spn]))

            if use_modality_tokens:
                text_spans = [" ".join((MODALITY_TOKEN_TEXT, text_span)) for text_span in text_spans]
                hubert_spans = [" ".join((MODALITY_TOKEN_SPEECH, hubert_span)) for hubert_span in hubert_spans]

            mm_spans = (text_spans, hubert_spans) if start_with_text else (hubert_spans, text_spans)
            interleaved_segment = " ".join(
                [span for spans in zip_longest(*mm_spans) for span in spans if span is not None]
            )
            g.write(json.dumps({MEGATRON_TEXT_KEY: interleaved_segment}) + "\n")

    LOGGER.info(f"Wrote {count_lines(output_jsonl)} lines to {output_jsonl!s}")
    return output_jsonl


if __name__ == "__main__":
    args = parse_args()
    interleave_dataset(**vars(args))
