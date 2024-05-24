#!/usr/bin/env python

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import SEED
from sklearn.cluster import MiniBatchKMeans


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__name__)


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate([feat[offsets[i] : offsets[i] + lengs[i]] for i in indices], axis=0)
        LOGGER.info((f"sampled {nsample} utterances, {len(sampled_feat)} frames " f"from shard {rank}/{nshard}"))
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [load_feature_shard(feat_dir, split, nshard, r, percent) for r in range(nshard)],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat_dir,
    split,
    nshard,
    km_path: Path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    km_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    LOGGER.info("total intertia: %.5f", inertia)
    LOGGER.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("km_path", type=Path)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--percent", default=-1, type=float, help="sample a subset; -1 for all")
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
