#!/usr/bin/env python

from __future__ import annotations

import os
import platform
import string
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional._alignment import TokenSpan


TERMINAL_WIDTH: int = os.get_terminal_size().columns
pprint = partial(pprint, width=TERMINAL_WIDTH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Specify audio backend - handle bug in PyTorch 2.1.0 https://github.com/facebookresearch/demucs/issues/570
TORCHAUDIO_BACKEND = "soundfile" if platform.system() == "Darwin" else None


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--speech-file", required=True, type=Path, help="Path to input audio file containing speech")
    parser.add_argument("--transcript", required=True, type=str, help="Text transcript of speech audio file")
    parser.add_argument("--emission-figure-savepath", required=True, type=Path, help="Path to save emissions plot")
    parser.add_argument(
        "--segments-output-dir", required=True, type=Path, help="Path to save word-level audio segments"
    )
    parser.add_argument("--verbose", action="store_true", help="Path to save emissions plot")
    return parser.parse_args()


def remove_punct(s: str) -> str:
    return s.translate(str.maketrans({key: None for key in string.punctuation}))


def plot_emissions(emission: Tensor, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()
    fig.savefig(output_path)


def align(emission: Tensor, tokens: list[int], device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    if device is None:
        device = emission.device
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)  # explicit default: blank=0
    alignments, scores = alignments.squeeze(0), scores.squeeze(0)  # squeeze batch dimension
    scores = scores.exp()  # convert from log probs to probs
    return alignments, scores


def unflatten_list(flattened_list: list[Any], lengths: list[int]) -> list[list[Any]]:
    """Unflatten a list of elements given the lengths of the original (inner nested) lists"""
    if len(flattened_list) != sum(lengths):
        raise ValueError(f"Length of flattened_list {len(flattened_list)} not equal to sum of lengths {sum(lengths)}")
    i = 0
    nested = []
    for length in lengths:
        nested.append(flattened_list[i : i + 1])
        i += length
    return nested


def compute_weighted_score(spans: list[TokenSpan]):
    """Compute average score weighted by the span length"""
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def segment_audio_by_spans(
    waveform: Tensor,
    token_spans: list[TokenSpan],
    num_frames: int,
    transcript: str,
    sr: int,
    output_path: Path,
    verbose: bool = True,
) -> Tensor:
    """Segment and save an audio clip given a sequence of TokenSpans for a given word of the transcript"""
    ratio = waveform.size(1) / num_frames
    start_sample = int(ratio * token_spans[0].start)
    end_sample = int(ratio * token_spans[-1].end)
    if verbose:
        weighted_score = compute_weighted_score(token_spans)
        print(f"{transcript} [weighted score: {weighted_score:.3f}]: {start_sample / sr:.3f}s - {end_sample / sr:.3f}s")
    segment = waveform[:, start_sample:end_sample]
    sf.write(output_path, segment.cpu().numpy(), sr)
    return segment


def main(args) -> None:

    if args.verbose:
        print(f"{torch.__version__ = }")
        print(f"{torchaudio.__version__ = }")
        print(f"{DEVICE = }")

    waveform, _ = torchaudio.load(args.speech_file, backend="soundfile")  # waveform shape is torch.Size([1, 54400])
    transcript_normalized = remove_punct(args.transcript).lower().split()

    mms_fa_bundle = torchaudio.pipelines.MMS_FA
    model = mms_fa_bundle.get_model(with_star=False).to(DEVICE)
    sample_rate: int = mms_fa_bundle._sample_rate  # type: ignore - torchaudio.pipelines.Wav2Vec2Bundle types as float

    with torch.inference_mode():
        emission, _ = model(waveform.to(DEVICE))

    args.emission_figure_savepath.parent.mkdir(exist_ok=True, parents=True)
    plot_emissions(emission.squeeze(0), args.emission_figure_savepath)

    # Tokenise the transcript
    fa_labels = mms_fa_bundle.get_labels(star=None)
    fa_dict = mms_fa_bundle.get_dict(star=None)

    for k, v in fa_dict.items():
        print(f"{k}: {v}")

    tokenized_transcript = [fa_dict[c] for word in transcript_normalized for c in word]
    print(f"{tokenized_transcript = }")

    aligned_tokens, alignment_scores = align(emission, tokenized_transcript, DEVICE)

    for i, (aligned_token, alignment_score) in enumerate(zip(aligned_tokens, alignment_scores)):
        # NOTE Alignments expressed in emission frames (i.e. not original waveform timestamps)
        print(f"{i:3d}:\t{aligned_token:2d} [{fa_labels[aligned_token]}], {alignment_score:.2f}")

    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

    print("Token\tTime\tScore")
    for s in token_spans:
        print(f"{fa_labels[s.token]}\t{s.start:3d}, {s.end:3d}\t{s.score:.2f}")

    word_spans = unflatten_list(token_spans, [len(word) for word in transcript_normalized])

    print(transcript_normalized)
    args.segments_output_dir.mkdir(exist_ok=True, parents=True)

    assert len(word_spans) == len(transcript_normalized), "Differing numbers of word span lists and words"
    n_words = len(str(len(word_spans)))
    num_frames = emission.size(1)

    for i, (char_spans, transcribed_word) in enumerate(zip(word_spans, transcript_normalized)):
        output_path = (args.segments_output_dir / f"{i:0{n_words}}_{transcribed_word}").with_suffix(".wav")
        segment_audio_by_spans(
            waveform, char_spans, num_frames, transcribed_word, sample_rate, output_path, args.verbose
        )


if __name__ == "__main__":
    main(parse_args())
