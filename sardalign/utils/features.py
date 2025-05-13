import logging
from pathlib import Path
from typing import Callable

import fairseq
import torch
import torch.nn.functional as F
import tqdm
from fairseq.data.audio.audio_utils import get_features_or_waveform
from npy_append_array import NpyAppendArray
from torch import Tensor

from sardalign.constants import SAMPLING_FREQ
from sardalign.utils import mls_id_to_path, read_jsonl


LOGGER = logging.getLogger(__name__)


class HubertLengthError(ValueError):
    pass


def get_shard_range(tot: int, nshard: int, rank: int) -> tuple[int, int]:
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    LOGGER.info(f"rank {rank} of {nshard}, process {end-start} " f"({start}-{end}) out of {tot}")
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


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1_600_000):
        (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        LOGGER.info(f"TASK CONFIG:\n{self.task.cfg}")
        LOGGER.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path: str, ref_len: int | None = None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path: str, ref_len: int | None = None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            # NOTE for pre-trained HuBERT (large; embed_dim = 1_024) cfg.normalize is True; and
            # NOTE self.task is a fairseq.tasks.hubert_pretraining.HubertPretrainingTask instance
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


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
    LOGGER.info("finished successfully")


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
            raise HubertLengthError(f"Audio length {x.size(1)} exceeds maximum length {self.max_len}")
        with torch.no_grad():
            if self.task.cfg.normalize:  # True for pre-trained HuBERT Large (w/ embed_dim = 1_024)
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)
            feat, _ = self.model.extract_features(source=x, padding_mask=None, mask=False, output_layer=self.layer)
        return feat.squeeze(0)
