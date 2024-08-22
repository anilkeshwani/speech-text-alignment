#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import write_jsonl


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--json", type=Path, required=True, help="Path to the Gigaspeech JSON file")
    parser.add_argument("--data-dir", type=Path, required=True, help="GigaSpeech directory; top-level, not /audio")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for segmented audio")
    parser.add_argument("--output-audio-format", type=str, default="WAV", help="Output format for segmented audio")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path to the output JSON lines file. "
        "Provided to specify different output directory and output JSON lines output paths",
    )
    parser.add_argument("--head", type=int, default=-1, help="Head audio files to preprocess; -1 for all")
    args = parser.parse_args()

    if not args.data_dir.is_absolute():
        raise ValueError("data-dir must be an absolute path")  # NOTE require absolute paths for now

    if args.head == 0:
        raise ValueError("Invalid to specify 0 samples for head")

    if args.output_audio_format not in sf.available_formats():
        raise ValueError(f"Invalid output audio format. Must be one of {list(sf.available_formats().keys())}")

    if args.output_jsonl is None:
        head_suffix = f"_head_{args.head}" if args.head != -1 else ""
        stem = args.json.stem + head_suffix
        args.output_jsonl = (args.output_dir / stem).with_suffix(".jsonl")

    return args


def main(args: Namespace):
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")
    with open(args.json) as f:
        metadata = json.load(f)  # NOTE Slow. Alternatives include jsonstreamer and ijson
    LOGGER.info(f"Read GigaSpeech input JSON from {args.json}")

    # NOTE top-level keys 'dataset', 'description', 'language', 'version' just contain basic info:
    # {
    #     "dataset": "GigaSpeech",
    #     "description": "GigaSpeech is a large English dataset which contains 10k hours of transcribed audio, and 33k "
    #     "hours of total audio, covering audio book, podcast and YouTube videos.",
    #     "language": "EN",
    #     "version": "v1.0.0",
    # }
    # keys for the individual audio segments:
    # NOTE "Audio keys" for individual audio segments:
    # {
    #     'codec', 'channels', 'url', 'path', 'duration', 'aid', 'source', 'subsets', 'sample_rate',
    #     'md5', 'segments', 'size', 'format', 'category', 'title', 'speaker'
    # }
    # NOTE Example of contents of an "audio" entry excluding the 'segments' item
    # {
    #     k: metadata["audios"][0][k]
    #     for k in [
    #         "title",
    #         "url",
    #         "path",
    #         "aid",
    #         "source",
    #         "format",
    #         "sample_rate",
    #         "codec",
    #         "channels",
    #         "md5",
    #         "duration",
    #         "size",
    #         "speaker",
    #         "category",
    #         "subsets",
    #     ]
    # }
    # {
    #     "title": "Check Cashing Stores",
    #     "url": "https://99percentinvisible.org/episode/episode-18-check-cashing-stores-download-embed/download",
    #     "path": "audio/podcast/P0001/POD0000000001.opus",
    #     "aid": "POD0000000001",
    #     "source": "podcast",
    #     "format": "opus",
    #     "sample_rate": 16000,
    #     "codec": "s16le",
    #     "channels": 1,
    #     "md5": "a1eb19f5bc68d4af71f65a3e8d700343",
    #     "duration": 629.931,
    #     "size": 2555888,
    #     "speaker": "N/A",
    #     "category": "Arts",
    #     "subsets": ["{XL}", "{L}"],
    # }
    # NOTE 'segments' item in audio entries are lists where each list element is a dictionary containing the following:
    # metadata['audios'][0]['segments'][0].keys()
    # -> dict_keys(['sid', 'speaker', 'begin_time', 'end_time', 'text_raw', 'text_tn', 'subsets'])
    # metadata['audios'][0]['segments'][0]
    # {
    #     "sid": "POD0000000001_S0000008",
    #     "speaker": "N/A",
    #     "begin_time": 159.0,
    #     "end_time": 167.52,
    #     "text_raw": "",
    #     "text_tn": "DOUGLAS MCGRAY IS GOING TO BE OUR GUIDE YOU WALK THROUGH THE DOOR <COMMA> YOU "
    #     "SEE THE RED CARPETING <COMMA> YOU SEE SOMEONE IN A SUIT <PERIOD> THEY MAY BE GREETING YOU <PERIOD>",
    #     "subsets": ["{XL}"],
    # }

    if args.head == -1:
        args.head = len(metadata["audios"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gigaspeech: list[dict] = []
    for ad in tqdm(metadata["audios"][: args.head]):
        audio_path = Path(ad["path"])
        wave, sr = librosa.load(args.data_dir / audio_path, sr=None)
        assert sr == 16_000, f"Sampling rate should be 16,000 but got {sr}"
        segments_subdir = audio_path.with_suffix("")
        (args.output_dir / segments_subdir).mkdir(parents=True, exist_ok=True)
        for segment in tqdm(ad["segments"]):
            segment_wave = wave[round(segment["begin_time"] * sr) : round(segment["end_time"] * sr)]
            segment_id = segment["sid"]
            assert segment_id.startswith(ad["aid"]), f"Segment ID without audio ID prefix: {segment_id} in {audio_path}"
            aid, per_segment_id = segment_id.split("_")  # fails if sid not of the form "{aid}_{sid}"
            segment_path = (segments_subdir / per_segment_id).with_suffix(audio_path.suffix)
            sample_dict = {
                "path": str(segment_path),
                "sid": segment_id,
                "speaker": segment.get("speaker"),
                "begin_time": segment.get("begin_time"),
                "end_time": segment.get("end_time"),
                "text_raw": segment.get("text_raw"),
                "text_tn": segment.get("text_tn"),
                "subsets": segment.get("subsets"),
            }
            gigaspeech.append(sample_dict)
            sf.write(args.output_dir / segment_path, segment_wave, samplerate=sr, format=args.output_audio_format)
    write_jsonl(args.output_jsonl, gigaspeech)
    LOGGER.info(f"Wrote {len(gigaspeech)} samples to {args.output_dir!s}")
    LOGGER.info(f"Wrote output JSON lines manifest to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
