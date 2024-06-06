#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchaudio
from sardalign.align import get_alignments
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import STAR_TOKEN
from sardalign.dump_km_label import ApplyKmeans
from sardalign.utils import count_lines, echo_environment_info, get_device, mls_id_to_path, read_jsonl
from sardalign.utils.align import get_span_times, get_spans, load_mms_aligner_model_and_dict
from sardalign.utils.features import SimpleHubertFeaturizer
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
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to input JSON lines file")
    parser.add_argument("--audio-dir", type=Path, help="Path to audio directory")
    parser.add_argument("--suffix", type=str, default=".flac", help="File extension for audio files")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Output path for JSON lines with alignments and HuBERT speech tokens",
    )
    parser.add_argument("--head", type=int, default=None, help="Use only head samples of the dataset; for testing")
    # MMS Aligner parameters
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("--text-key", type=str, default="transcript", help="Key of text field in JSON lines manifest")
    parser.add_argument("--normalized-key", type=str, default="normalized", help="Key for normalized tokens")
    parser.add_argument("--uroman-key", type=str, default="uroman", help="Key for uroman tokens in JSON lines manifest")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=None,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
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
    if args.out_jsonl is None:
        args.out_jsonl = args.jsonl.with_stem(args.jsonl.stem + "_aligned_hubert")
    return args


def main(args):
    device = get_device(args.device)
    echo_environment_info(torch, torchaudio, device)

    if args.out_jsonl.exists():
        raise FileExistsError(f"Existing output JSON lines file at {args.out_jsonl!s}")

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

    LOGGER.info(f"Writing output to {args.out_jsonl!s}")

    tokens_s: list[list[str]] = [
        s[args.text_key].strip().split(args.token_delimiter) for s in tqdm(dataset, desc="Tokenization")
    ]
    norm_tokens_s: list[list[str]] = [s[args.normalized_key] for s in dataset]
    uroman_tokens_s: list[list[str]] = [s[args.uroman_key] for s in dataset]

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

    with open(args.out_jsonl, "x") as f:  # flushes buffer every ~150 lines on testing
        for sample, tokens, norm_tokens, uroman_tokens in tqdm(
            zip(dataset, tokens_s, norm_tokens_s, uroman_tokens_s),
            desc="Aligning and encoding HuBERT tokens",
            total=len(dataset),
        ):
            audio_path = mls_id_to_path(sample["ID"], audio_dir=args.audio_dir, suffix=args.suffix)
            segments, stride_ms, wave = get_alignments(
                audio_path, uroman_tokens, mms_aligner_model, mms_aligner_dict, args.use_star, device
            )
            spans = get_spans(uroman_tokens, segments)
            assert len(tokens) == len(
                spans
            ), f"Length mismatch: len(spans) = {len(spans)} vs len(tokens) = {len(tokens)}"
            sample |= {"speech_tokens": kmeans(hubert_featurizer(wave)).tolist()}
            sample |= {"alignment": {token: get_span_times(span, stride_ms) for (token, span) in zip(tokens, spans)}}
            f.write(json.dumps(sample) + "\n")

    LOGGER.info(f"Wrote {count_lines(args.out_jsonl)} lines to {args.out_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
