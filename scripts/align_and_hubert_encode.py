#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from sardalign.align import get_alignments
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import (
    ALIGNMENT_END_TIME_KEY,
    ALIGNMENT_START_TIME_KEY,
    NORMALIZED_KEY,
    SPEECH_TOKENS_KEY,
    STAR_TOKEN,
    TEXT_KEY_DEFAULT,
    TOKENIZED_KEY,
    UROMAN_KEY,
)
from sardalign.dump_km_label import ApplyKmeans
from sardalign.utils import count_lines, echo_environment_info, get_device, mls_id_to_path, read_jsonl
from sardalign.utils.align import get_span_times, get_spans, load_mms_aligner_model_and_dict
from sardalign.utils.features import SimpleHubertFeaturizer


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to input JSON lines file")
    # MLS-related arguments
    # NOTE ids-not-paths, audio-dir and suffix are used by MLS manifests which retain IDs not audio paths
    parser.add_argument(
        "--ids-not-paths",
        action="store_true",
        help="Flag indicating manifests contain IDs not paths. Used for MLS."
        "IDs will be transformed to path via mls_id_to_path using the --audio-dir and --suffix",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".flac",
        help="File extension for audio files. Used if JSON lines provides IDs not paths.",
    )
    # NOTE retain audio-dir argument after removing MLS ID-specific functionality as relative paths are still useful
    #      given that datasets may need to be arbitrarily relocated on disk
    parser.add_argument(
        "--audio-dir",
        default=None,
        type=Path,
        help="Path to root audio directory. Used if paths in JSON lines manifest are relative to this directory",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output path for JSON lines with alignments and HuBERT speech tokens",
    )
    parser.add_argument("--head", type=int, default=None, help="Use only head samples of the dataset; for testing")
    # MMS Aligner parameters
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    # TODO remove the keys are argparse arguments and move them into constants, like speech tokens and alignment keys
    parser.add_argument(
        "--text-key", type=str, default=TEXT_KEY_DEFAULT, help="Key of text field in JSON lines manifest"
    )
    parser.add_argument("--use-star", action="store_true", help="Use star at the start of transcript")
    # HuBERT parameters
    parser.add_argument("--hubert-ckpt-path", type=str, required=True, help="Path to HuBERT checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="Layer of the HuBERT model to use")
    # k-means parameters
    parser.add_argument("--km-ckpt-path", type=Path, required=True, help="Path to k-means (joblib) serialised model")
    # Hardware parameters
    parser.add_argument("--device", type=str, default=None, help="Torch device; in string format")
    args = parser.parse_args()

    if args.ids_not_paths:
        if args.audio_dir is None:
            raise ValueError("Must specify audio directory (--audio-dir) with --ids-not-paths")
        LOGGER.info(f"Got --ids-not-paths. Audio directory set to {args.audio_dir} and audio extension {args.suffix}")

    if args.output_jsonl is None:
        args.output_jsonl = args.jsonl.with_stem(args.jsonl.stem + "_aligned_hubert")
    return args


def main(args):
    device = get_device(args.device)
    echo_environment_info(torch, torchaudio, device)

    if args.output_jsonl.exists():
        raise FileExistsError(f"Existing output JSON lines file at {args.output_jsonl!s}")

    if args.head is not None:
        dataset: list[dict] = []
        with open(args.jsonl) as f:
            for i, line in enumerate(f):
                if i == args.head:
                    break
                dataset.append(json.loads(line))
    else:
        dataset = read_jsonl(args.jsonl)
    LOGGER.info(f"Read {len(dataset)} lines from {args.jsonl!s}")

    LOGGER.info(f"Writing output to {args.output_jsonl!s}")

    tokens_s: list[list[str]] = [s[TOKENIZED_KEY] for s in dataset]
    norm_tokens_s: list[list[str]] = [s[NORMALIZED_KEY] for s in dataset]
    uroman_tokens_s: list[list[str]] = [s[UROMAN_KEY] for s in dataset]

    for i, (tokens, norm_tokens, uroman_tokens) in enumerate(zip(tokens_s, norm_tokens_s, uroman_tokens_s)):
        if (len(tokens) != len(norm_tokens)) or (len(tokens) != len(uroman_tokens)):
            raise ValueError(f"Found incongruous number of tokens in line {i + 1} reading from manifest {args.jsonl!s}")

    # load MMS alignment model and respective dictionary
    mms_aligner_model, mms_aligner_dict = load_mms_aligner_model_and_dict()
    mms_aligner_model = mms_aligner_model.to(device)

    if args.use_star:
        mms_aligner_dict[STAR_TOKEN] = len(mms_aligner_dict)
        tokens_s = [[STAR_TOKEN] + tokens for tokens in tokens_s]
        norm_tokens_s = [[STAR_TOKEN] + norm_tokens for norm_tokens in norm_tokens_s]
        uroman_tokens_s = [[STAR_TOKEN] + uroman_tokens for uroman_tokens in uroman_tokens_s]

    # Load HuBERT model via featurizer and k-means model
    hubert_featurizer = SimpleHubertFeaturizer(ckpt_path=args.hubert_ckpt_path, layer=args.layer, device=device)
    kmeans = ApplyKmeans(args.km_ckpt_path)

    with open(args.output_jsonl, "x") as f:  # flushes buffer every ~150 lines on testing
        for sample, tokens, norm_tokens, uroman_tokens in tqdm(
            zip(dataset, tokens_s, norm_tokens_s, uroman_tokens_s),
            desc="Aligning and encoding HuBERT tokens",  # TODO Make HuBERT encoding optional
            total=len(dataset),
        ):
            if args.ids_not_paths:
                audio_path = mls_id_to_path(sample["ID"], audio_dir=args.audio_dir, suffix=args.suffix)
            else:
                audio_path = Path(sample["path"])
            segments, stride_ms, wave = get_alignments(
                audio_path, uroman_tokens, mms_aligner_model, mms_aligner_dict, args.use_star, device
            )
            spans = get_spans(uroman_tokens, segments)
            span_times: list[tuple[float, float]] = [get_span_times(span, stride_ms) for span in spans]
            assert len(tokens) == len(spans), f"Length mismatch: len(spans)={len(spans)} vs len(tokens)={len(tokens)}"
            # TODO Make HuBERT encoding optional - simple if required here + CLI argument e.g. --encode-hubert
            sample |= {SPEECH_TOKENS_KEY: kmeans(hubert_featurizer(wave)).tolist()}
            sample |= {
                ALIGNMENT_START_TIME_KEY: [span[0] for span in span_times],
                ALIGNMENT_END_TIME_KEY: [span[1] for span in span_times],
            }
            f.write(json.dumps(sample) + "\n")

    LOGGER.info(f"Wrote {count_lines(args.output_jsonl)} lines to {args.output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
