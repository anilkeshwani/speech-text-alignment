import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from math import ceil

import torch
from torchaudio.models import wav2vec2_model, Wav2Vec2Model

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import CTC_ALIGNMENT_MLING_UROMAN_DICT_PATH, CTC_ALIGNMENT_MLING_UROMAN_MODEL_PATH


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)

# iso codes with specialized rules in uroman
special_isos_uroman = "ara, bel, bul, deu, ell, eng, fas, grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid".split(
    ","
)
special_isos_uroman = [i.strip() for i in special_isos_uroman]


def normalize_uroman(text: str) -> str:
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts: list[str], uroman_root_dir: str, iso: str | None = None):
    tf = tempfile.NamedTemporaryFile()
    tf2 = tempfile.NamedTemporaryFile()
    with open(tf.name, "w") as f:
        for t in norm_transcripts:
            f.write(t + "\n")

    assert os.path.exists(f"{uroman_root_dir}/uroman.pl"), "uroman not found"
    cmd = f"perl {uroman_root_dir}/uroman.pl"
    if iso in special_isos_uroman:
        cmd += f" -l {iso} "
    cmd += f" < {tf.name} > {tf2.name}"
    os.system(cmd)
    outtexts = []
    with open(tf2.name) as f:
        for line in f:
            line = " ".join(line.strip())
            line = re.sub(r"\s+", " ", line).strip()
            outtexts.append(line)
    assert len(outtexts) == len(norm_transcripts)
    uromans = []
    for ot in outtexts:
        uromans.append(normalize_uroman(ot))
    return uromans


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def load_mms_aligner_model_and_dict(
    model_path_name: str = CTC_ALIGNMENT_MLING_UROMAN_MODEL_PATH,
    dict_path_name: str = CTC_ALIGNMENT_MLING_UROMAN_DICT_PATH,
) -> tuple[Wav2Vec2Model, dict[str, int]]:
    LOGGER.info("Downloading model and dictionary...")
    if os.path.exists(model_path_name):
        LOGGER.info("Model path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
            model_path_name,
        )
        assert os.path.exists(model_path_name)
    state_dict = torch.load(model_path_name, map_location="cpu")

    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(state_dict)
    model.eval()

    if os.path.exists(dict_path_name):
        LOGGER.info("Dictionary path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt",
            dict_path_name,
        )
        assert os.path.exists(dict_path_name)
    dictionary = {}
    with open(dict_path_name) as f:
        dictionary = {l.strip(): i for i, l in enumerate(f.readlines())}

    return model, dictionary


def get_spans(tokens, segments):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    sil = "<blank>"
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == "<blank>"
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == "<blank>":
            continue
        assert seg.label == ltr
        if (ltr_idx) == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start : end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == sil:
                pad_start = prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end) / 2)
                span = [Segment(sil, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == sil:
                pad_end = (
                    next_seg.end if (idx == len(intervals) - 1) else math.floor((next_seg.start + next_seg.end) / 2)
                )
                span = span + [Segment(sil, span[-1].end, pad_end)]
        spans.append(span)
    return spans


def get_span_times(span: list[Segment], stride_ms: float) -> tuple[float, float]:
    """
    Calculate the start and end times of a span in seconds.

    Args:
        span (list[Segment]): A list of Segment objects representing the span.
        stride_ms (float): The stride in milliseconds.

    Returns:
        tuple[float, float]: A tuple containing the start and end times of the span in seconds.
    """
    seg_start_idx = span[0].start
    seg_end_idx = span[-1].end
    audio_start_sec = seg_start_idx * stride_ms / 1000
    audio_end_sec = seg_end_idx * stride_ms / 1000
    return audio_start_sec, audio_end_sec


def times_to_hubert_idxs(times: tuple[float, float], sampling_rate: int, downsampling_ratio: int) -> tuple[int, int]:
    span_start_sec, span_end_sec = times
    start_idx = int(span_start_sec * sampling_rate / downsampling_ratio)
    end_idx = int(ceil(span_end_sec * sampling_rate / downsampling_ratio))
    return start_idx, end_idx
