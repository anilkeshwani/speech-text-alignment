import json
import logging
import os
import platform
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
import torch
import torchaudio
import torchaudio.functional as F
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import EMISSION_INTERVAL, SAMPLING_FREQ
from sardalign.text_normalization import text_normalize
from sardalign.utils import echo_environment_info, get_device
from sardalign.utils.align import get_spans, get_uroman_tokens, load_model_dict, merge_repeats, time_to_frame
from torch import Tensor


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)

DEVICE = get_device()
TORCHAUDIO_BACKEND = "soundfile" if platform.system() == "Darwin" else None


def generate_emissions(model: torchaudio.models.Wav2Vec2Model, audio_file: str | Path) -> tuple[Tensor, float]:
    waveform, _ = torchaudio.load(audio_file, backend=TORCHAUDIO_BACKEND)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)  # output from bash soxi -D "$audio_file"; e.g. 15.180000
    assert total_duration is not None
    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ
    context: float = EMISSION_INTERVAL * 0.1
    emissions_arr = []
    with torch.inference_mode():
        t_seconds = 0
        while t_seconds < total_duration:  # NOTE usually lasts 1 iteration - EMISSION_INTERVAL is 30s out-of-the-box
            segment_start_time, segment_end_time = (t_seconds, t_seconds + EMISSION_INTERVAL)
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[:, int(SAMPLING_FREQ * input_start_time) : int(SAMPLING_FREQ * (input_end_time))]
            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)
            emissions_ = emissions_[emission_start_frame - offset : emission_end_frame - offset, :]
            emissions_arr.append(emissions_)
            t_seconds += EMISSION_INTERVAL
    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)
    stride_ms = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)  # milliseconds
    return emissions, stride_ms


def get_alignments(
    audio_file: str | Path,
    uroman_tokens: list[str],
    model: torchaudio.models.Wav2Vec2Model,
    dictionary: dict[str, int],
    use_star: bool,
):
    # generate emissions: log prob distributions of uroman tokens per Wav2Vec2 output frame (usually 320x downsampling)
    emissions, stride_ms = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)  # add star entry to dist with zero prob
    # force alignment
    if uroman_tokens:
        token_indices = [dictionary[c] for c in " ".join(uroman_tokens).split(" ") if c in dictionary]  # TODO ??
    else:
        LOGGER.warning(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]

    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)

    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    path, _ = F.forced_align(emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank)
    path = path.squeeze().to("cpu").tolist()

    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride_ms


def main(args):
    echo_environment_info(torch, torchaudio, DEVICE)
    assert not os.path.exists(args.outdir), f"Error: Output path exists already {args.outdir}"
    with open(args.text_filepath) as f:
        transcripts: list[str] = [line.strip() for line in f]
    print(f"Read {len(transcripts)} lines from {args.text_filepath}")
    norm_transcripts: list[str] = [text_normalize(line.strip(), args.lang) for line in transcripts]
    tokens: list[str] = get_uroman_tokens(norm_transcripts, args.uroman_path, args.lang)
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts
    segments, stride = get_alignments(
        args.audio_filepath,
        tokens,
        model,
        dictionary,
        args.use_star,
    )
    spans = get_spans(tokens, segments)  # get spans of each line in input text file
    os.makedirs(args.outdir)
    with open(f"{args.outdir}/manifest.json", "w") as f:
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end
            output_file = f"{args.outdir}/segment{i}.flac"
            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000
            tfm = sox.Transformer()
            tfm.trim(audio_start_sec, audio_end_sec)
            tfm.build_file(args.audio_filepath, output_file)
            sample = {
                "audio_start_sec": audio_start_sec,
                "audio_filepath": str(output_file),
                "duration": audio_end_sec - audio_start_sec,
                "text": t,
                "normalized_text": norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            f.write(json.dumps(sample) + "\n")
    return segments, stride


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Align and segment long audio files")
    parser.add_argument("-a", "--audio_filepath", type=Path, help="Path to input audio file")
    parser.add_argument("-t", "--text_filepath", type=str, help="Path to input text file ")
    parser.add_argument("-l", "--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("-u", "--uroman_path", type=str, default="eng", help="Location to uroman/bin")
    parser.add_argument("-s", "--use_star", action="store_true", help="Use star at the start of transcript")
    parser.add_argument("-o", "--outdir", type=str, help="Output directory to store segmented audio files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
