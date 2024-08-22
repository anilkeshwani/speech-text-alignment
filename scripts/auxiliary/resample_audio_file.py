#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
from pathlib import Path

import librosa
import soundfile as sf


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", required=True, type=Path, help="Path to input audio file")
    parser.add_argument("-o", "--output", default=None, type=Path, help="Path to output resampled audio file")
    parser.add_argument("-t", "--target-sample-rate", required=True, type=int, help="Target sampling rate")
    args = parser.parse_args()
    if args.output is None:
        args.output = (args.file.parent / (args.file.stem + str(args.target_sample_rate))).with_suffix(args.file.suffix)
    return args


def main(args: Namespace) -> None:
    y, sr = librosa.load(args.file, sr=None)
    y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=args.target_sample_rate)
    sf.write(args.output, y_resampled, samplerate=args.target_sample_rate)


if __name__ == "__main__":
    main(parse_args())
