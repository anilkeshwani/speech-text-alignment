# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import tqdm
from npy_append_array import NpyAppendArray
from sardalign.utils import mls_id_to_path, read_jsonl


if TYPE_CHECKING:
    from sardalign.dump_hubert_feature import HubertFeatureReader
else:
    from typing import Any

    HubertFeatureReader = Any


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")


def get_shard_range(tot: int, nshard: int, rank: int) -> tuple[int, int]:
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(f"rank {rank} of {nshard}, process {end-start} " f"({start}-{end}) out of {tot}")
    return start, end


def get_path_iterator(tsv, nshard: int, rank: int):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]

        def iterate():
            for line in lines:
                _ = line.split("\t")
                subpath, nsample = (*_, None) if len(_) == 1 else _
                yield f"{root}/{subpath}", int(nsample) if nsample is not None else nsample

    return iterate, len(lines)


def get_mls_path_iterator(
    jsonl: Path, audio_dir: Path, nshard: int, rank: int, suffix: str = ".flac"
) -> tuple[Callable, int]:
    lines: list[dict] = read_jsonl(jsonl)
    start, end = get_shard_range(len(lines), nshard, rank)
    lines = lines[start:end]

    def iterate():
        for line in lines:
            audio_path = mls_id_to_path(line["ID"], audio_dir, suffix)
            n_samples = line.get("n_samples")
            n_samples = int(n_samples) if n_samples is not None else n_samples
            yield audio_path, n_samples

    return iterate, len(lines)


def dump_feature(
    reader: HubertFeatureReader, generator: Callable, num: int, split: str | int, nshard: int, rank: int, feat_dir: Path
):
    iterator = generator()

    feat_path = feat_dir / f"{split!s}_{rank}_{nshard}.npy"
    leng_path = feat_dir / f"{split!s}_{rank}_{nshard}.len"

    feat_dir.mkdir(exist_ok=True)

    if feat_path.exists():
        raise FileExistsError(f"Existing features NumPy array at {feat_path!s}")

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")
