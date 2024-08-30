#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace

from datasets import Dataset, NamedSplit
from huggingface_hub import HfApi

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

HF_DATASETS_BASE_URL: str = "https://huggingface.co/datasets"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # NOTE on path_or_paths: CLI accepts single path - argument name is for convenient passing to Dataset.from_json
    parser.add_argument("path_or_paths", type=str, help="Path to JSON lines file")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="A namespace (user or an organization) and a repo name separated by a /",
    )
    parser.add_argument("--split", type=str, default=None, help="Split name to be assigned to the dataset.")
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help="Set to `'dataset'` or `'space'` if uploading to a dataset or space, `None` or `'model'` if uploading to "
        "a model. Defaults to `'dataset'`.",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--private", action="store_true", help="Whether the model repo should be private.")
    parser.add_argument("--keep-in-memory", action="store_true", help="Whether to copy the data in-memory.")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes when downloading and generating the dataset locally. "
        "This is helpful if the dataset is made of multiple files. Multiprocessing is disabled by default.",
    )
    parser.add_argument(
        "-m",
        "--commit-message",
        type=str,
        default=None,
        help="Message to commit while pushing. Will default to 'Upload dataset'.",
    )
    args = parser.parse_args()
    return args


def main(
    path_or_paths: str,
    split: str | None,
    cache_dir: str | None,
    keep_in_memory: bool,
    num_proc: int | None,
    repo_id: str,
    repo_type: str,
    private: bool,
    commit_message: str,
) -> None:
    hfapi = HfApi()
    wai = hfapi.whoami()
    username = wai["name"]
    LOGGER.info(f"Logged in as user {username} ({wai['fullname']}. Email: {wai['email']})")
    if "/" not in repo_id:
        repo_id = "/".join((username, repo_id))
        LOGGER.info(f"No repo namespace specified. Inferring from username: {repo_id}")
    if split is not None:
        split = NamedSplit(split)
    ds = Dataset.from_json(
        path_or_paths,
        split=split,
        cache_dir=cache_dir,  # type: ignore
        keep_in_memory=keep_in_memory,
        num_proc=num_proc,
    )
    LOGGER.info(f"Built HF dataset with {len(ds)} samples ({type(ds)})")
    hfapi.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    LOGGER.info(f"Created HF {repo_type} repo: {repo_id}")
    LOGGER.info(f"Repo visibility: " + ("Private" if private else "Public"))
    # NOTE push_to_hub: split defaults to self.split; private does not affect existing repo
    ds.push_to_hub(repo_id, commit_message=commit_message)
    LOGGER.info(f"Completed upload of dataset to {repo_id} (split: {ds.split._name}). ")
    LOGGER.info(f"Link: {HF_DATASETS_BASE_URL}/{repo_id}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
