#!/usr/bin/env python

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
import torch
import torchaudio

from sardalign.align_and_segment import get_alignments
from sardalign.align_utils import get_spans, get_uroman_tokens, load_model_dict
from sardalign.constants import STAR_TOKEN
from sardalign.text_normalization import text_normalize
from sardalign.utils import echo_environment_info, get_device, ljspeech_id_to_path, read_jsonl


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-j", "--jsonl", required=True, type=Path, help="Path to input JSON Lines file")
    parser.add_argument(
        "--ljspeech-wavs-dir",
        default=Path("/media/scratch/anilkeshwani/data/LJSpeech-1.1/wavs_16000"),
        type=Path,
        help="Path to directory containing wavs. Option specific to the LJSpeech dataset, whose manifest contains IDs",
    )
    parser.add_argument("-o", "--outdir", required=True, type=Path, help="Output directory for segmented audio files")
    parser.add_argument("-l", "--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("-u", "--uroman-path", default=None, type=Path, help="Location to uroman/bin")
    parser.add_argument("-s", "--use-star", action="store_true", help="Use star at the start of transcript")
    parser.add_argument(
        "--transcript-stem-suffix", action="store_true", help="Append transcript span to output audio filenames"
    )
    parser.add_argument("--sample", default=None, type=int, help="Use a sample of the dataset for testing purposes")
    args = parser.parse_args()
    if args.uroman_path is None:
        args.uroman_path = Path(__file__).parents[1] / "submodules" / "uroman" / "bin"
    return args


def main(args):
    DEVICE = get_device()
    TOKEN_DELIMITER_SPLIT: str | None = None  # None (default to str.split) splits on any whitespace
    TEXT_KEY: str = "normalized_transcription"

    echo_environment_info(torch, torchaudio, DEVICE)

    args.outdir.mkdir(parents=True, exist_ok=False)

    dataset = read_jsonl(args.jsonl)
    if args.sample is not None:
        dataset = dataset[: args.sample]
    print(f"Read {len(dataset)} lines from {args.jsonl}")

    transcripts_s: list[list[str]] = [s[TEXT_KEY].strip().split(TOKEN_DELIMITER_SPLIT) for s in dataset]
    norm_transcripts_s = [[text_normalize(token, args.lang) for token in transcripts] for transcripts in transcripts_s]
    tokens_s = [get_uroman_tokens(nt, args.uroman_path, args.lang) for nt in norm_transcripts_s]

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)

    if args.use_star:
        dictionary[STAR_TOKEN] = len(dictionary)
        tokens_s = [[STAR_TOKEN] + tokens for tokens in tokens_s]
        transcripts_s = [[STAR_TOKEN] + transcripts for transcripts in transcripts_s]
        norm_transcripts_s = [[STAR_TOKEN] + norm_transcripts for norm_transcripts in norm_transcripts_s]

    ljspeech_id_s = [sd["ID"] for sd in dataset]
    assert len(tokens_s) == len(transcripts_s) == len(norm_transcripts_s) == len(ljspeech_id_s)

    segments_s, stride_s = [], []

    for tokens, transcripts, norm_transcripts, lj_id in zip(tokens_s, transcripts_s, norm_transcripts_s, ljspeech_id_s):
        assert len(tokens) == len(transcripts) == len(norm_transcripts), "Inconsistent tokens after norm/uroman G2P"
        audio_path = ljspeech_id_to_path(lj_id, wavs_dir=args.ljspeech_wavs_dir)
        segments, stride = get_alignments(audio_path, tokens, model, dictionary, args.use_star)

        spans = get_spans(tokens, segments)

        outdir_segment = args.outdir / lj_id
        outdir_segment.mkdir()
        with open(outdir_segment / "manifest.json", "x") as f:
            for i, t in enumerate(transcripts):
                span = spans[i]
                seg_start_idx = span[0].start
                seg_end_idx = span[-1].end

                audio_start_sec = seg_start_idx * stride / 1000
                audio_end_sec = seg_end_idx * stride / 1000

                transcript_stem_suffix = f"_{t}" if args.transcript_stem_suffix else ""
                output_file = (outdir_segment / f"segment_{i}{transcript_stem_suffix}").with_suffix(".flac")

                tfm = sox.Transformer()
                tfm.trim(audio_start_sec, audio_end_sec)
                tfm.build_file(audio_path, output_file)

                sample = {
                    "audio_start_sec": audio_start_sec,
                    "audio_filepath": str(output_file),
                    "duration": audio_end_sec - audio_start_sec,
                    "text": t,
                    "normalized_text": norm_transcripts[i],
                    "uroman_tokens": tokens[i],
                }
                f.write(json.dumps(sample) + "\n")

        segments_s.append(segments)
        stride_s.append(stride)

    return segments_s, stride_s


if __name__ == "__main__":
    args = parse_args()
    main(args)
