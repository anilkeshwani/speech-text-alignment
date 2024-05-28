#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
import torch
import torchaudio
from sardalign.align_and_segment import get_alignments
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import STAR_TOKEN
from sardalign.utils import echo_environment_info, get_device, mls_id_to_path, read_jsonl
from sardalign.utils.align import get_spans, load_model_dict
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
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for segmented audio files")
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
    parser.add_argument("--device", type=str, default=None, help="Device for ")
    parser.add_argument("--head", type=int, default=None, help="Use only head samples of the dataset; for testing")
    args = parser.parse_args()
    return args


def main(args):
    device = get_device(args.device)
    echo_environment_info(torch, torchaudio, device)

    args.out_dir.mkdir(parents=True, exist_ok=False)

    dataset = read_jsonl(args.jsonl)
    if args.head is not None:
        dataset = dataset[: args.head]
    LOGGER.info(f"Read {len(dataset)} lines from {args.jsonl}")

    tokens_s: list[list[str]] = [
        s[args.text_key].strip().split(args.token_delimiter) for s in tqdm(dataset, desc="Tokenization")
    ]
    norm_tokens_s: list[list[str]] = [s[args.normalized_key] for s in dataset]
    uroman_tokens_s: list[list[str]] = [s[args.uroman_key] for s in dataset]
    file_id_s = [sd["ID"] for sd in dataset]

    for i, (tokens, norm_tokens, uroman_tokens) in enumerate(zip(tokens_s, norm_tokens_s, uroman_tokens_s)):
        if (len(tokens) != len(norm_tokens)) or (len(tokens) != len(uroman_tokens)):
            raise ValueError(f"Found incongruous number of tokens in line {i + 1} reading from manifest {args.jsonl!s}")

    model, dictionary = load_model_dict()
    model = model.to(device)

    if args.use_star:
        dictionary[STAR_TOKEN] = len(dictionary)
        tokens_s = [[STAR_TOKEN] + tokens for tokens in tokens_s]
        norm_tokens_s = [[STAR_TOKEN] + norm_tokens for norm_tokens in norm_tokens_s]
        uroman_tokens_s = [[STAR_TOKEN] + uroman_tokens for uroman_tokens in uroman_tokens_s]

    segments_s, stride_s = [], []

    for file_id, tokens, norm_tokens, uroman_tokens in zip(file_id_s, tokens_s, norm_tokens_s, uroman_tokens_s):
        audio_path = mls_id_to_path(file_id, audio_dir=args.audio_dir, suffix=args.suffix)
        segments, stride = get_alignments(audio_path, uroman_tokens, model, dictionary, args.use_star)
        spans = get_spans(uroman_tokens, segments)

        outdir_segment = args.out_dir / file_id
        outdir_segment.mkdir()
        with open(outdir_segment / "manifest.json", "x") as f:
            for i, token in enumerate(tokens):
                span = spans[i]
                seg_start_idx = span[0].start
                seg_end_idx = span[-1].end

                audio_start_sec = seg_start_idx * stride / 1000
                audio_end_sec = seg_end_idx * stride / 1000

                output_file = (outdir_segment / f"segment_{i}").with_suffix(".flac")

                tfm = sox.Transformer()
                tfm.trim(audio_start_sec, audio_end_sec)
                tfm.build_file(audio_path, output_file)

                sample = {
                    "audio_start_sec": audio_start_sec,
                    "audio_filepath": str(output_file),
                    "duration": audio_end_sec - audio_start_sec,
                    "text": token,
                    "normalized_text": norm_tokens[i],
                    "uroman_tokens": uroman_tokens[i],
                }
                f.write(json.dumps(sample) + "\n")

        segments_s.append(segments)
        stride_s.append(stride)

    return segments_s, stride_s


if __name__ == "__main__":
    args = parse_args()
    main(args)
