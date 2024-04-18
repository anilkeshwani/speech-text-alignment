#!/usr/bin/env python

from __future__ import annotations

import platform
from pathlib import Path
from pprint import pprint

import IPython
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional._alignment import TokenSpan


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Specify audio backend - handle bug in PyTorch 2.1.0 https://github.com/facebookresearch/demucs/issues/570
TORCHAUDIO_BACKEND = "soundfile" if platform.system() == "Darwin" else None
EMISSION_FIGURE_SAVEPATH = Path("./figures/emissions.png")


def plot_emission(emission: Tensor, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path)


def align(emission: Tensor, tokens: list[int]) -> tuple[Tensor, Tensor]:
    targets = torch.tensor([tokens], dtype=torch.int32, device=DEVICE)
    alignments, scores = F.forced_align(emission, targets, blank=0)  # explicit default: blank=0
    alignments, scores = alignments.squeeze(0), scores.squeeze(0)  # squeeze batch dimension
    scores = scores.exp()  # convert from log probs to probs
    return alignments, scores


def unflatten(token_spans: list[TokenSpan], lengths: list[int]) -> list[list[TokenSpan]]:
    assert len(token_spans) == sum(lengths)
    i = 0
    nested_spans = []
    for length in lengths:
        nested_spans.append(token_spans[i : i + 1])
        i += length
    return nested_spans


def weighted_score(spans):
    """Compute average score weighted by the span length"""
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def main() -> None:

    print(torch.__version__)
    print(torchaudio.__version__)

    print(DEVICE)

    SPEECH_FILE = Path("./data/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

    waveform, _ = torchaudio.load(SPEECH_FILE, backend="soundfile")
    TRANSCRIPT = "i had that curiosity beside me at this moment".split()

    bundle = torchaudio.pipelines.MMS_FA

    model = bundle.get_model(with_star=False).to(DEVICE)

    with torch.inference_mode():
        emission, _ = model(waveform.to(DEVICE))

    pprint(emission)
    pprint(type(emission))
    pprint(emission.shape)

    plot_emission(emission.squeeze(0), EMISSION_FIGURE_SAVEPATH)

    # Tokenise the transcript
    LABELS = bundle.get_labels(star=None)
    DICTIONARY = bundle.get_dict(star=None)

    print(f"{type(LABELS) = }")
    print(f"{type(DICTIONARY) = }")

    for k, v in DICTIONARY.items():
        print(f"{k}: {v}")

    TOKENIZED_TRANSCRIPT = [DICTIONARY[c] for word in TRANSCRIPT for c in word]
    print(f"{TOKENIZED_TRANSCRIPT = }")

    aligned_tokens, alignment_scores = align(emission, TOKENIZED_TRANSCRIPT)

    for i, (aligned_token, alignment_score) in enumerate(zip(aligned_tokens, alignment_scores)):
        print(f"{i:3d}:\t{aligned_token:2d} [{LABELS[aligned_token]}], {alignment_score:.2f}")

    # NOTE The alignment is expressed in the frame cordinate of the emission (different from the original waveform)

    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

    print("Token\tTime\tScore")
    for s in token_spans:
        print(f"{LABELS[s.token]}\t{s.start:3d}, {s.end:3d}\t{s.score:.2f}")

    word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])

    print(type(word_spans))
    print(type(word_spans[0]))
    print(type(word_spans[0][0]))


if __name__ == "__main__":
    main()
