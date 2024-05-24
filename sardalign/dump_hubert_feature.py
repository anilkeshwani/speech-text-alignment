import logging
import os
import sys
from pathlib import Path
from pprint import pformat

import fairseq
import torch
import torch.nn.functional as F
from fairseq.data.audio.audio_utils import get_features_or_waveform
from sardalign.utils.features import dump_feature, get_mls_path_iterator


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1_600_000):
        (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
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


def main(
    jsonl: Path,
    audio_dir: Path,
    # NOTE ckpt_path can't be a `Path` as `fairseq.checkpoint_utils.get_maybe_sharded_checkpoint_filename` (line 383)
    #      calls filename.replace(".pt", suffix + ".pt") with ckpt_path passed as the filename
    ckpt_path: str,
    layer: int,
    nshard: int,
    rank: int,
    feat_dir: Path,
    max_chunk: int,
    suffix: str,
):
    split = jsonl.stem
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    generator, num = get_mls_path_iterator(jsonl, audio_dir=audio_dir, nshard=nshard, rank=rank, suffix=suffix)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to JSON lines manifest file")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing MLS audios")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to pre-trained HuBERT checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="BERT layer whose embeddings to use as features")
    parser.add_argument("--nshard", type=int, default=1, help="Number of shards for parallel processing")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--feat-dir", type=Path, required=True, help="Output directory for HuBERT features")
    parser.add_argument("--max-chunk", type=int, default=1_600_000, help="Maximum audio chunk length in samples")
    parser.add_argument("--suffix", type=str, default=".flac", help="File extension for audio files")
    args = parser.parse_args()
    logger.info(pformat(vars(args), sort_dicts=False))

    main(**vars(args))
