#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
import torch
import torchaudio

from sardalign.align import get_alignments
from sardalign.text_normalization import text_normalize
from sardalign.utils import echo_environment_info, get_device
from sardalign.utils.align import get_spans, get_uroman_tokens, load_mms_aligner_model_and_dict


def main(args):
    device = get_device(args.device)
    echo_environment_info(torch, torchaudio, device)
    assert not os.path.exists(args.outdir), f"Error: Output path exists already {args.outdir}"
    with open(args.text_filepath) as f:
        transcripts: list[str] = [line.strip() for line in f]
    print(f"Read {len(transcripts)} lines from {args.text_filepath}")
    norm_transcripts: list[str] = [text_normalize(line.strip(), args.lang) for line in transcripts]
    tokens: list[str] = get_uroman_tokens(norm_transcripts, args.uroman_path, args.lang)
    model, dictionary = load_mms_aligner_model_and_dict()
    model = model.to(device)
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts
    segments, stride = get_alignments(args.audio_filepath, tokens, model, dictionary, args.use_star, device)
    spans = get_spans(tokens, segments)  # get spans of each line in input text file
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


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Align and segment long audio files")
    parser.add_argument("-a", "--audio_filepath", type=Path, help="Path to input audio file")
    parser.add_argument("-t", "--text_filepath", type=str, help="Path to input text file ")
    parser.add_argument("-l", "--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("-u", "--uroman_path", type=str, default="eng", help="Location to uroman/bin")
    parser.add_argument("-s", "--use_star", action="store_true", help="Use star at the start of transcript")
    parser.add_argument("-o", "--outdir", type=str, help="Output directory to store segmented audio files")
    parser.add_argument("--device", type=str, default=None, help="Torch device; in string format")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
