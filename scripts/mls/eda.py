#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import TEXT_KEY
from sardalign.utils import read_jsonl
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--token-delimiter", type=str, default=None)
    parser.add_argument("--hist-path", type=Path, required=True)
    return parser.parse_args()


def compute_sequence_length_distribution(jsonl_path: Path, token_delimiter: str | None) -> NDArray:
    dataset = read_jsonl(jsonl_path)
    sequence_lens = [len(s[TEXT_KEY].strip().split()) for s in tqdm(dataset, desc="Computing sequence lengths")]
    return np.array(sequence_lens)


def main(jsonl_path: Path, token_delimiter: str | None, hist_path: Path):
    sequence_lens = compute_sequence_length_distribution(jsonl_path, token_delimiter)
    _quantiles = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    quantiles = {k: v for k, v in zip(_quantiles, np.quantile(sequence_lens, _quantiles))}
    LOGGER.info(f"Summary statistics for {jsonl_path!s}:")
    LOGGER.info(f"Sequence length quantiles: \n{pformat(quantiles)}")
    LOGGER.info(f"Mean sequence length: {np.mean(sequence_lens):,.2f}")
    LOGGER.info(f"Std dev. of sequence length: {np.std(sequence_lens):,.2f}")
    sns.set_theme()
    a4_dims = (16, 12)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.histplot(ax=ax, data=sequence_lens, stat="count", bins=50)
    ax.set(xlabel="Sequence Length", ylabel="Count")
    max_seq_len = max(sequence_lens)
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.95 * max_seq_len),  # TODO labels are plotted too low
            f"{height:,}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10,
        )
    plt.savefig(hist_path)
    LOGGER.info(f"Wrote histogram to {hist_path!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args.jsonl_path, args.token_delimiter, args.hist_path)
