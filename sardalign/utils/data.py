import logging
from pprint import pformat
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


def get_stratified_sample(
    data: DataFrame,
    sample_size: int,
    strata_label: str,
    shuffle: bool,
    verbose: bool,
    logger: logging.Logger,
    seed: int | None = None,
) -> tuple[DataFrame, int]:
    prng = np.random.default_rng(seed)
    N = len(data)
    if sample_size >= N:
        raise ValueError("Sample size should be less than number of samples")
    logger.info(f"Obtaining stratified sample of size {sample_size} (from {N} total samples)")
    criterion = data[strata_label]
    strata, s_invs, s_cnts = np.unique(criterion, return_inverse=True, return_counts=True)
    n_strata = len(strata)
    logger.info(f"Number of strata: {n_strata}")
    if sample_size < n_strata:
        logger.warning(
            f"Sample size ({sample_size}) is smaller than number of strata ({n_strata}). "
            "The current implementation deterministically takes the least populated strata even when passing --shuffle"
            ", which selects samples from within each stratum randomly and does not randomise the selected strata."
        )
    idxs_cnts_desc = np.argsort(s_cnts)[::-1]
    speaker_distribution_desc = {s: c for s, c in zip(strata[idxs_cnts_desc], s_cnts[idxs_cnts_desc])}
    if verbose:
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
