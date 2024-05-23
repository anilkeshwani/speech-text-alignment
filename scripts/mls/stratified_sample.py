#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat, pprint
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from sardalign.constants import SEED
from sardalign.utils import parse_arg_int_or_float
from sklearn.model_selection import train_test_split


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


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
    if args.output_jsonl is None:
        stem = args.transcripts_jsonl.stem + f"_stratified_sample_{args.sample}"
        args.output_jsonl = args.transcripts_jsonl.with_stem(stem)
    return args


def get_stratified_sample(
    data: DataFrame,
    sample_size: int | float,
    strata_label: str,
    shuffle: bool,
    verbose: bool,
    seed: int | None = None,
):
    prng = np.random.default_rng(seed)
    N = len(data)
    if isinstance(sample_size, float):
        sample_size = int(sample_size * N)
    if sample_size >= N:
        raise ValueError("Sample size should be less than number of samples")
    print(f"Obtaining stratified sample of size {sample_size} (from {N} total samples)")
    criterion = data[strata_label]
    strata, s_invs, s_cnts = np.unique(criterion, return_inverse=True, return_counts=True)
    n_strata = len(strata)
    print(f"Number of strata: {n_strata}")
    idxs_cnts_desc = np.argsort(s_cnts)[::-1]
    speaker_distribution_desc = {s: c for s, c in zip(strata[idxs_cnts_desc], s_cnts[idxs_cnts_desc])}
    print(f"Speaker distribution (descending):\n{pformat(speaker_distribution_desc, sort_dicts=False)}")
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
    _ = {s: len(idxs) for s, idxs in ss_idxs_selected.items()}
    _ = {s: _[s] for s in speaker_distribution_desc}
    pprint(_)
    n_total_samples_taken = sum(len(_) for _ in ss_idxs_selected.values())
    assert n_total_samples_taken == sample_size, f"{n_total_samples_taken} samples taken, {sample_size} requested"
    exit()
    ss_idxs_selected = np.concatenate(ss_idxs_selected.values())
    return data.loc[ss_idxs_selected]


def main(args):
    if args.output_jsonl.exists():
        raise FileExistsError(f"File already present at {args.output_jsonl}")
    # transcripts df contains an "ID" column of the form 4800_10003_000000 which is {speaker}_{book}_{audio}
    transcripts = pd.read_json(args.transcripts_jsonl, lines=True, orient="records", dtype={"ID": str})
    # Create a new speaker column by extracting the number before the first underscore from the "ID" column
    transcripts["speaker"] = transcripts["ID"].str.split("_").str[0].astype(int)
    # get_stratified_sample(transcripts, args.sample, "speaker", args.shuffle, )
    get_stratified_sample(
        transcripts, args.sample, strata_label="speaker", shuffle=args.shuffle, verbose=args.verbose, seed=args.seed
    )
    exit()
    train, _ = train_test_split(transcripts, train_size=args.sample, stratify=transcripts["speaker"], random_state=SEED)
    train: pd.DataFrame
    train.drop(columns="speaker", inplace=True)
    camios.drop(columns="speaker", inplace=True)
    stratified_sample = pd.concat([ss, camios], axis="index")
    with open(args.output_jsonl, "x") as f:
        f.write(stratified_sample.to_json(orient="records", lines=True, force_ascii=args.force_ascii))
    logger.info(f"Wrote {len(stratified_sample)} lines to {args.output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
