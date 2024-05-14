#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
from sardalign.utils import mls_id_to_path, write_jsonl
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True, help="Path to the MLS transcripts.txt file.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing the audio files.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to the output JSON lines file.")
    parser.add_argument("--audio-ext", type=str, default=".flac", help="File extension of the audio files.")
    parser.add_argument("--field-delimiter", type=str, default="\t", help="Field delimiter used in transcripts file.")
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to take; -1 for all")
    return parser.parse_args()


def main(args: Namespace):
    if args.output_jsonl.exists():
        raise FileExistsError(f"JSON lines output file already exists at {args.output_jsonl}")
    transcript_lines: list[str] = args.transcripts.read_text().splitlines()
    if args.sample > 0:
        transcript_lines = transcript_lines[: args.sample]

    mls: list[dict] = []
    for line in tqdm(transcript_lines):
        mls_id, transcript = line.strip().split(args.field_delimiter)
        audio_path = mls_id_to_path(mls_id, args.audio_dir, suffix=args.audio_ext)
        audio_file_info = sox.file_info.info(audio_path)
        mls.append(
            {
                "ID": mls_id,
                "transcript": transcript,
                "duration": audio_file_info["duration"],
                "sample_rate": audio_file_info["sample_rate"],
                "num_samples": audio_file_info["num_samples"],
            }
        )
    write_jsonl(args.output_jsonl, mls)
    print(f"Wrote {len(mls)} samples to {args.output_jsonl!s}")


if __name__ == "__main__":
    main(parse_args())
