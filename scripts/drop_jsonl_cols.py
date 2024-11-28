#!/usr/bin/env python

# TODO Can be moved to utils/ subdirectory of scripts/ when appropriate

import json
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON lines file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON lines file")
    parser.add_argument(
        "--columns", type=str, nargs="+", required=True, help="Columns to drop from the input JSON lines file"
    )
    return parser.parse_args()


def main(args):
    columns = set(args.columns)

    with open(args.input, "r") as infile, open(args.output, "w") as outfile:
        for line in infile:
            record = json.loads(line)
            filtered_record = {k: v for k, v in record.items() if k not in columns}
            outfile.write(json.dumps(filtered_record) + "\n")

    print(f"Columns {columns} dropped. Output saved to {args.output}.")


if __name__ == "__main__":
    main(parse_args())
