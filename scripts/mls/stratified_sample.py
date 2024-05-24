#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sardalign.constants import SEED
from sardalign.utils import count_lines, parse_arg_int_or_float


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--transcripts-jsonl", required=True, type=Path, help="transcripts.jsonl")
    parser.add_argument("--output-jsonl", default=None, type=Path, help="Output path for stratified sample jsonl")
    parser.add_argument("--sample", required=True, type=parse_arg_int_or_float, help="transcripts.txt")
    parser.add_argument("--force-ascii", action="store_true", help="Escape non-ASCII characters in JSON lines output")
    parser.add_argument("--shuffle", action="store_true", help="Whether to randomise the sample from each stratum")
    parser.add_argument("--seed", default=SEED, type=int, help="Random seed")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def get_integer_sample_size(sample_size: int | float, N: int) -> int:
    if not isinstance(sample_size, (int, float)):
        raise TypeError(f"sample_size should be one of int or float but got {type(sample_size)}")
    if isinstance(args.sample, float):
        sample_size = int(args.sample * N)
    return sample_size


def get_stratified_sample(
    data: DataFrame,
    sample_size: int,
    strata_label: str,
    shuffle: bool,
    verbose: bool,
    seed: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[DataFrame, int]:
    prng = np.random.default_rng(seed)
    N = len(data)
    if sample_size >= N:
        raise ValueError("Sample size should be less than number of samples")
    if verbose and logger is not None:
        logger.info(f"Obtaining stratified sample of size {sample_size} (from {N} total samples)")
    criterion = data[strata_label]
    strata, s_invs, s_cnts = np.unique(criterion, return_inverse=True, return_counts=True)
    n_strata = len(strata)
    if verbose and logger is not None:
        logger.info(f"Number of strata: {n_strata}")
    idxs_cnts_desc = np.argsort(s_cnts)[::-1]
    speaker_distribution_desc = {s: c for s, c in zip(strata[idxs_cnts_desc], s_cnts[idxs_cnts_desc])}
    if verbose and logger is not None:
        logger.info(f"Speaker distribution (descending):\n{pformat(speaker_distribution_desc, sort_dicts=False)}")
    s_idxs = np.argsort(s_invs, kind="stable")  # stable so the head of a stratum corresponds to samples' original order
    ss_idxs: list[NDArray] = np.split(s_idxs, np.cumsum(s_cnts)[:-1])
    if shuffle:
        [prng.shuffle(ss) for ss in ss_idxs]  # in-place
    assert sum(len(ss) for ss in ss_idxs) == N
    assert all(len(ss_idx) == s_cnt for ss_idx, s_cnt in zip(ss_idxs, s_cnts))
    ss_idxs_selected: dict[Any, NDArray] = {}
    ss_idxs_asc = [ss_idxs[i] for i in idxs_cnts_desc[::-1]]
    samples_to_take = sample_size
    for i, (stratum, ss_idx) in enumerate(zip(reversed(speaker_distribution_desc), ss_idxs_asc)):
        desired_samples_stratum = samples_to_take // (n_strata - i)
        ss_idxs_selected[stratum] = ss_idx[:desired_samples_stratum]
        samples_to_take -= len(ss_idxs_selected[stratum])
    assert sum(len(_) for _ in ss_idxs_selected.values()) == sample_size
    ss_idxs_selected = np.concatenate(list(ss_idxs_selected.values()))
    stratified_sample = data.loc[ss_idxs_selected]
    assert len(stratified_sample) == sample_size
    return stratified_sample, N


def main(args):
    N = count_lines(args.transcripts_jsonl)
    sample_size = get_integer_sample_size(args.sample, N)
    if args.output_jsonl is None:
        stem = args.transcripts_jsonl.stem + f"_stratified_sample_{sample_size}"
        args.output_jsonl = args.transcripts_jsonl.with_stem(stem)
    if args.output_jsonl.exists():
        raise FileExistsError(f"File already present at {args.output_jsonl}")
    # transcripts JSON lines contains an "ID" column of the form 4800_10003_000000 which is {speaker}_{book}_{audio}
    transcripts = pd.read_json(args.transcripts_jsonl, lines=True, orient="records", dtype={"ID": str})
    transcripts["speaker"] = transcripts["ID"].str.split("_").str[0].astype(int)  # speaker ID from ID column
    stratified_sample, _N = get_stratified_sample(
        transcripts,
        sample_size,
        strata_label="speaker",
        shuffle=args.shuffle,
        verbose=args.verbose,
        seed=args.seed,
        logger=LOGGER,
    )
    assert N == _N
    stratified_sample.drop(columns="speaker", inplace=True)
    if args.verbose:
        LOGGER.info(stratified_sample.head())
    with open(args.output_jsonl, "x") as f:
        f.write(stratified_sample.to_json(orient="records", lines=True, force_ascii=args.force_ascii))
    LOGGER.info(f"Wrote {sample_size} lines to {args.output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
