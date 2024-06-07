#!/usr/bin/env python

import logging
import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sox
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import TEXT_KEY
from sardalign.utils import mls_id_to_path, read_jsonl
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


TEXT_SEQ_LENS_HIST_PNG_NAME = "mls_strat_sample_seq_lengths_histogram.png"
AUDIO_DURATIONS_HIST_PNG_NAME = "mls_strat_sample_audio_lengths_histogram.png"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--token-delimiter", type=str, default=None)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--hist-dir", type=Path, required=True)
    return parser.parse_args()


def eda_length_dist(
    sequence_lens: np.ndarray,
    probs: list[float],
    hist_path: Path,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (16, 12),
    label_height_scale: float = 0.05,
    file: Path | None = None,
):
    seq_len_quantiles = {k: v for k, v in zip(probs, np.quantile(sequence_lens, probs))}
    if file is not None:
        LOGGER.info(f"Summary statistics for {file!s}:")
    LOGGER.info(f"Sequence length quantiles (tokens): \n{pformat(seq_len_quantiles)}")
    LOGGER.info(f"Mean sequence length (tokens): {np.mean(sequence_lens):,.2f}")
    LOGGER.info(f"Std dev. of sequence length (tokens): {np.std(sequence_lens):,.2f}")

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(ax=ax, data=sequence_lens, stat="count", bins=50)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    max_seq_len = max(sequence_lens)
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (label_height_scale * max_seq_len),  # TODO labels are plotted too low
            f"{height:,}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10,
        )
    plt.savefig(hist_path)
    LOGGER.info(f"Wrote histogram of sequence lengths (in tokens) to {hist_path!s}")


def duration_lambda(samples: list[dict], audio_dir: Path) -> list[float | None]:
    durs: list[float | None] = []
    for sample in tqdm(samples):
        durs.append(sox.file_info.duration(mls_id_to_path(sample["ID"], audio_dir=audio_dir)))
    return durs


def main(jsonl_path: Path, token_delimiter: str | None, audio_dir: Path, hist_dir: Path):
    dataset = read_jsonl(jsonl_path)
    probs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]

    # token sequence length distribution
    sequence_lens = np.array(
        [len(s[TEXT_KEY].strip().split(token_delimiter)) for s in tqdm(dataset, desc="Computing text sequence lengths")]
    )
    eda_length_dist(
        sequence_lens,
        probs,
        hist_path=hist_dir / TEXT_SEQ_LENS_HIST_PNG_NAME,
        xlabel="Text Sequence Length / tokens",
        ylabel="Count",
        label_height_scale=0.5,
        file=jsonl_path,
    )

    # audio length distribution
    n_processes = max(mp.cpu_count() // 2, 1)
    chunk_size = len(dataset) // n_processes
    dataset_chunks = [dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)]
    with mp.Pool(processes=n_processes) as pool:
        audio_durations_s = pool.starmap(duration_lambda, [(dc, audio_dir) for dc in dataset_chunks])
    audio_durations = np.array([audio_dur for audio_durs in audio_durations_s for audio_dur in audio_durs])
    eda_length_dist(
        audio_durations,
        probs,
        hist_path=hist_dir / AUDIO_DURATIONS_HIST_PNG_NAME,
        xlabel="Audio Duration / s",
        ylabel="Count",
        file=jsonl_path,
    )

    # speaker distribution
    # speakers = [s["ID"].split("_")[0]]  # MLS IDs are of form {speaker}_{book}_{audio} e.g. 4800_10003_000000
    # Counter()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
