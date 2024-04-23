#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
import torch
import torchaudio
import torchaudio.functional as F
from sardalign.align_and_segment import generate_emissions, get_alignments
from sardalign.align_utils import get_spans, get_uroman_tokens, load_model_dict, merge_repeats, time_to_frame
from sardalign.constants import EMISSION_INTERVAL, SAMPLING_FREQ
from sardalign.text_normalization import text_normalize
from sardalign.utils import read_jsonl


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_TOKEN_DELIMITER = None  # None (default to str.split) splits on any whitespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path, help="Path to input JSON Lines file")
    parser.add_argument("--outdir", required=True, type=Path, help="Path to the output directory")
    parser.add_argument("--lang", default="en-US", type=str, help="Path to the output directory")
    return parser.parse_args()


def main(args):
    if args.outdir.exits():
        raise FileExistsError(f"Output path exists already {args.outdir}")

    dataset = read_jsonl(args.jsonl)
    print(f"Read {len(dataset)} lines from {args.jsonl}")

    norm_transcripts = [
        text_normalize(line["normalized_transcription"].strip().split(SPLIT_TOKEN_DELIMITER), args.lang)
        for line in transcripts
    ]
    tokens = get_uroman_tokens(norm_transcripts, args.uroman_path, args.lang)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(
        args.audio_filepath,
        tokens,
        model,
        dictionary,
        args.use_star,
    )
    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)

    os.makedirs(args.outdir)
    with open(f"{args.outdir}/manifest.json", "w") as f:
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end

            output_file = f"{args.outdir}/segment{i}.flac"

            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000

            tfm = sox.Transformer()
            tfm.trim(audio_start_sec, audio_end_sec)
            tfm.build_file(args.audio_filepath, output_file)

            sample = {
                "audio_start_sec": audio_start_sec,
                "audio_filepath": str(output_file),
                "duration": audio_end_sec - audio_start_sec,
                "text": t,
                "normalized_text": norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            f.write(json.dumps(sample) + "\n")

    return segments, stride
