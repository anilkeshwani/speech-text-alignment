#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from itertools import zip_longest
from math import ceil
from pathlib import Path

import fairseq
import sox
import torch
import torch.nn.functional as F
import torchaudio
from sardalign.align import get_alignments
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import SAMPLING_FREQ, STAR_TOKEN
from sardalign.dump_km_label import ApplyKmeans
from sardalign.utils import echo_environment_info, get_device, mls_id_to_path, read_jsonl
from sardalign.utils.align import get_spans, load_mms_aligner_model_and_dict
from sardalign.utils.features import HubertFeatureReader
from torch import Tensor
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to input JSON lines file")
    parser.add_argument("--audio-dir", type=Path, help="Path to audio directory")
    parser.add_argument("--suffix", type=str, default=".flac", help="File extension for audio files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for segmented audio files")
    parser.add_argument("--head", type=int, default=None, help="Use only head samples of the dataset; for testing")
    # MMS Aligner parameters
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("--text-key", type=str, default="transcript", help="Key of text field in JSON lines manifest")
    parser.add_argument("--normalized-key", type=str, default="normalized", help="Key for normalized tokens")
    parser.add_argument("--uroman-key", type=str, default="uroman", help="Key for uroman tokens in JSON lines manifest")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=None,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    parser.add_argument("--use-star", action="store_true", help="Use star at the start of transcript")
    # HuBERT parameters
    parser.add_argument("--hubert-ckpt-path", type=str, required=True, help="Path to HuBERT checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="Layer of the HuBERT model to use")
    # k-means parameters
    parser.add_argument("--km-ckpt-path", type=Path, required=True, help="Path to k-means (joblib) serialised model")
    # Hardware parameters
    parser.add_argument("--device", type=str, default=None, help="Torch device; in string format")
    args = parser.parse_args()
    return args


class SimpleHubertFeaturizer:
    def __init__(self, ckpt_path: str, layer: int, device: torch.device, max_len: int = 100 * SAMPLING_FREQ):
        """
        Initializes a new instance of the `SimpleHubertFeaturizer` class.

        Args:
            ckpt_path (str): The path to the checkpoint file.
            layer (int): The layer of the model to use.
            device (torch.device): The device to use for inference.
            max_len (int, optional): The maximum length of the input sequence. Defaults to 100 * SAMPLING_FREQ, or 100s.

        Returns:
            None
        """
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(device)
        self.task: fairseq.tasks.hubert_pretraining.HubertPretrainingTask = task
        self.layer = layer
        self.max_len = max_len

    def __call__(self, x: Tensor):
        """
        Extracts features from the given audio tensor using the pre-trained HuBERT model.

        Args:
            x (torch.Tensor): The audio tensor of shape (channel, time).

        Returns:
            torch.Tensor: The extracted features of shape (feature_dim,).

        Raises:
            ValueError: If the audio length exceeds the maximum length specified.
        """
        if x.size(1) > self.max_len:
            raise ValueError(f"Audio length {x.size(1)} exceeds maximum length {self.max_len}")
        with torch.no_grad():
            if self.task.cfg.normalize:  # True for pre-trained HuBERT Large (w/ embed_dim = 1_024)
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)
            feat, _ = self.model.extract_features(source=x, padding_mask=None, mask=False, output_layer=self.layer)
        return feat.squeeze(0)


def main(args):
    device = get_device(args.device)
    echo_environment_info(torch, torchaudio, device)

    args.out_dir.mkdir(parents=True, exist_ok=False)

    if args.head is not None:
        dataset: list[dict] = []
        with open(args.jsonl) as f:
            for i, line in enumerate(f):
                if i == args.head:
                    break
                dataset.append(json.loads(line))
    else:
        dataset = read_jsonl(args.jsonl)
    LOGGER.info(f"Read {len(dataset)} lines from {args.jsonl}")

    tokens_s: list[list[str]] = [
        s[args.text_key].strip().split(args.token_delimiter) for s in tqdm(dataset, desc="Tokenization")
    ]
    norm_tokens_s: list[list[str]] = [s[args.normalized_key] for s in dataset]
    uroman_tokens_s: list[list[str]] = [s[args.uroman_key] for s in dataset]
    file_id_s = [sd["ID"] for sd in dataset]

    for i, (tokens, norm_tokens, uroman_tokens) in enumerate(zip(tokens_s, norm_tokens_s, uroman_tokens_s)):
        if (len(tokens) != len(norm_tokens)) or (len(tokens) != len(uroman_tokens)):
            raise ValueError(f"Found incongruous number of tokens in line {i + 1} reading from manifest {args.jsonl!s}")

    # load MMS alignment model and respective dictionary
    mms_aligner_model, mms_aligner_dict = load_mms_aligner_model_and_dict()
    mms_aligner_model = mms_aligner_model.to(device)

    if args.use_star:
        mms_aligner_dict[STAR_TOKEN] = len(mms_aligner_dict)
        tokens_s = [[STAR_TOKEN] + tokens for tokens in tokens_s]
        norm_tokens_s = [[STAR_TOKEN] + norm_tokens for norm_tokens in norm_tokens_s]
        uroman_tokens_s = [[STAR_TOKEN] + uroman_tokens for uroman_tokens in uroman_tokens_s]

    # Load HuBERT model via featurizer
    hubert_featurizer = SimpleHubertFeaturizer(ckpt_path=args.hubert_ckpt_path, layer=args.layer, device=device)

    # Cross-check that the fairseq HuBERT feature reader provides the same features as the re-implementation
    hubert_feature_reader = HubertFeatureReader(ckpt_path=args.hubert_ckpt_path, layer=args.layer)

    # Load k-means model
    kmeans = ApplyKmeans(args.km_ckpt_path)

    segments_s, stride_ms_s = [], []

    for file_id, tokens, norm_tokens, uroman_tokens in zip(file_id_s, tokens_s, norm_tokens_s, uroman_tokens_s):
        audio_path = mls_id_to_path(file_id, audio_dir=args.audio_dir, suffix=args.suffix)
        segments, stride_ms, wave = get_alignments(
            audio_path, uroman_tokens, mms_aligner_model, mms_aligner_dict, args.use_star, device
        )
        spans = get_spans(uroman_tokens, segments)
        assert len(tokens) == len(spans), f"Length mismatch: len(spans) = {len(spans)} vs len(tokens) = {len(tokens)}"

        speech_tokens_segment = kmeans(hubert_featurizer(wave)).tolist()

        with open(outdir_segment / "manifest.json", "x") as f:
            for i, (token, span) in enumerate(zip(tokens, spans)):
                seg_start_idx = span[0].start
                seg_end_idx = span[-1].end
                audio_start_sec = seg_start_idx * stride_ms / 1000
                audio_end_sec = seg_end_idx * stride_ms / 1000

                sampled_start_idx = int(audio_start_sec * SAMPLING_FREQ)
                sampled_end_idx = int(ceil(audio_end_sec * SAMPLING_FREQ))
                trimmed_waveform = wave[:, sampled_start_idx:sampled_end_idx]
                hubert_features = hubert_featurizer(trimmed_waveform)
                speech_tokens = kmeans(hubert_features).tolist()

                sample = {
                    "audio_start_sec": audio_start_sec,
                    "duration": audio_end_sec - audio_start_sec,
                    "text": token,
                    "normalized_text": norm_tokens[i],
                    "uroman_tokens": uroman_tokens[i],
                    "speech_tokens": speech_tokens,
                }
                f.write(json.dumps(sample) + "\n")

        segments_s.append(segments)
        stride_ms_s.append(stride_ms)

    return segments_s, stride_ms_s
"""

if __name__ == "__main__":
    args = parse_args()
    main(args)
