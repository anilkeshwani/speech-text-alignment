#!/usr/bin/env python

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import sox
import torch
import torchaudio
from sardalign.align_and_segment import get_alignments
from sardalign.constants import PROJECT_ROOT, STAR_TOKEN
from sardalign.text_normalization import text_normalize
from sardalign.utils import echo_environment_info, get_device, mls_id_to_path, read_jsonl
from sardalign.utils.align import get_spans, get_uroman_tokens, load_model_dict
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to input JSON lines file")
    parser.add_argument("--audio-dir", type=Path, help="Path to audio directory")
    parser.add_argument("--suffix", type=str, default=".flac", help="File extension for audio files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for segmented audio files")
    parser.add_argument("--lang", type=str, default="eng", help="ISO code of the language")
    parser.add_argument("--text-key", type=str, default="transcript", help="Key of text field in JSON lines manifest")
    parser.add_argument(
        "--token-delimiter",
        type=str,
        default=None,
        help="Token delimiter as used by str.split; defaults to None, i.e. splits on any whitespace",
    )
    parser.add_argument("--uroman-path", type=Path, default=None, help="Location to uroman/bin")
    parser.add_argument("--use-star", action="store_true", help="Use star at the start of transcript")
    parser.add_argument(
        "--transcript-stem-suffix", action="store_true", help="Append transcript span to output audio filenames"
    )
    parser.add_argument("--device", type=str, default=None, help="Device for ")
    parser.add_argument("--head", type=int, default=None, help="Use only head samples of the dataset; for testing")
    args = parser.parse_args()
    if args.uroman_path is None:
        args.uroman_path = PROJECT_ROOT / "submodules" / "uroman" / "bin"
    return args


def main(args):
    device = get_device(args.device)
    echo_environment_info(torch, torchaudio, device)

    args.out_dir.mkdir(parents=True, exist_ok=False)

    dataset = read_jsonl(args.jsonl)
    if args.head is not None:
        dataset = dataset[: args.head]
    print(f"Read {len(dataset)} lines from {args.jsonl}")

    transcripts_s: list[list[str]] = []
    for s in tqdm(dataset, desc="Tokenizing dataset"):
        transcripts_s.append(s[args.text_key].strip().split(args.token_delimiter))
    norm_transcripts_s = []
    for transcripts in tqdm(transcripts_s, desc="Normalizing transcripts"):
        norm_transcripts_s.append([text_normalize(token, args.lang) for token in transcripts])
    uroman_tokens_s = []
    for nt in tqdm(norm_transcripts_s, desc="Getting uroman tokens for transcripts"):
        uroman_tokens_s.append(get_uroman_tokens(nt, args.uroman_path, args.lang))

    model, dictionary = load_model_dict()
    model = model.to(device)

    if args.use_star:
        dictionary[STAR_TOKEN] = len(dictionary)
        uroman_tokens_s = [[STAR_TOKEN] + tokens for tokens in uroman_tokens_s]
        transcripts_s = [[STAR_TOKEN] + transcripts for transcripts in transcripts_s]
        norm_transcripts_s = [[STAR_TOKEN] + norm_transcripts for norm_transcripts in norm_transcripts_s]

    file_id_s = [sd["ID"] for sd in dataset]
    assert len(uroman_tokens_s) == len(transcripts_s) == len(norm_transcripts_s) == len(file_id_s)

    segments_s, stride_s = [], []

    for tokens, transcripts, norm_transcripts, file_id in zip(uroman_tokens_s, transcripts_s, norm_transcripts_s, file_id_s):
        assert len(tokens) == len(transcripts) == len(norm_transcripts), "Inconsistent tokens after norm/uroman G2P"
        audio_path = mls_id_to_path(file_id, audio_dir=args.audio_dir, suffix=args.suffix)
        segments, stride = get_alignments(audio_path, tokens, model, dictionary, args.use_star)

        spans = get_spans(tokens, segments)

        outdir_segment = args.out_dir / file_id
        outdir_segment.mkdir()
        with open(outdir_segment / "manifest.json", "x") as f:
            for i, t in enumerate(transcripts):
                span = spans[i]
                seg_start_idx = span[0].start
                seg_end_idx = span[-1].end

                audio_start_sec = seg_start_idx * stride / 1000
                audio_end_sec = seg_end_idx * stride / 1000

                transcript_stem_suffix = f"_{t}" if args.transcript_stem_suffix else ""
                output_file = (outdir_segment / f"segment_{i}{transcript_stem_suffix}").with_suffix(".flac")

                tfm = sox.Transformer()
                tfm.trim(audio_start_sec, audio_end_sec)
                tfm.build_file(audio_path, output_file)

                sample = {
                    "audio_start_sec": audio_start_sec,
                    "audio_filepath": str(output_file),
                    "duration": audio_end_sec - audio_start_sec,
                    "text": t,
                    "normalized_text": norm_transcripts[i],
                    "uroman_tokens": tokens[i],
                }
                f.write(json.dumps(sample) + "\n")

        segments_s.append(segments)
        stride_s.append(stride)

    return segments_s, stride_s


if __name__ == "__main__":
    args = parse_args()
    main(args)
