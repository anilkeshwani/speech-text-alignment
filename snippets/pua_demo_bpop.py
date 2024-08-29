# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import random
import unicodedata
from itertools import count, repeat

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer


# the original private offset pointed to a location in the middle of the first PUA range, which was too small for our vocab size
# PRIVATE_OFFSET = 61440
PRIVATE_OFFSET = 983040


def load_model(path):
    sp_model = spm.SentencePieceProcessor(model_file=path)
    proto = sp_pb2_model.ModelProto()
    proto.ParseFromString(sp_model.serialized_model_proto())
    return sp_model, proto


def create_piece(string, score=0, special=False):
    new_piece = sp_pb2_model.ModelProto().SentencePiece()
    new_piece.piece = string
    new_piece.score = score

    if special:
        new_piece.type = 4  # will this line ever cause a problem?
    return new_piece


def extra_id(i):
    return "<extra_id_{}>".format(str(i))


def pua(i):
    private_char = chr(i + PRIVATE_OFFSET)
    assert unicodedata.category(private_char) == "Co"  # otherwise, it's not private
    return private_char


def extend(proto, new_types, scoring, special=False):
    if scoring == "bpe":
        get_score = count(proto.pieces[-1].score - 1, -1)
    else:
        get_score = repeat(0)

    # extend the model proto
    for new_type in new_types:
        new_piece = create_piece(new_type, next(get_score), special=special)
        proto.pieces.append(new_piece)

    # you can also update the vocab size in proto.trainer_spec, but it doesn't
    # seem to matter

    return proto


def save(proto, model_prefix):
    with open(model_prefix + ".model", "wb") as f:
        f.write(proto.SerializeToString())

    with open(model_prefix + ".vocab", "w") as f:
        for piece in proto.pieces:
            f.write("\t".join([piece.piece, str(int(piece.score))]) + "\n")


def extend_tokenizer(original, dsus, dsu_type, model_prefix, scoring):
    if dsu_type == "extra_id":
        dsu_func = extra_id
        special = True
    else:
        dsu_func = pua
        special = False  # the PUA approach allows DSUs to be treated as ordinary characters, which may be desirable

    model, proto = load_model(args.original)

    new_types = [dsu_func(i) for i in range(args.dsus)]
    proto = extend(proto, new_types, args.scoring, special=special)

    save(proto, args.model_prefix + "_" + dsu_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default=None, type=str)  # llama
    parser.add_argument("--dsus", type=int, default=5000)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--scoring", default="bpe", choices=["bpe", "none"])
    args = parser.parse_args()

    # extra id style:
    extend_tokenizer(args.original, args.dsus, "extra_id", args.model_prefix, args.scoring)

    # pua style:
    extend_tokenizer(args.original, args.dsus, "pua", args.model_prefix, args.scoring)

    # Do they produce the same tokenization for a given input?
    dsu_ix = [random.randrange(args.dsus) for i in range(5)]
    input_base = "the dog went to the park {}{}{}{}{}"

    input_extra_id = input_base.format(*[extra_id(i) for i in dsu_ix])
    input_pua = input_base.format(*[pua(i) for i in dsu_ix])  # unfortunately this is not trivial

    print(input_extra_id)
    print(input_pua)

    # load tokenizers (with the transformers wrapper for the heck of it)
    tok_extra_id = LlamaTokenizer(vocab_file=args.model_prefix + "_extra_id.model")
    output_extra_id = tok_extra_id(input_extra_id).input_ids

    tok_pua = LlamaTokenizer(vocab_file=args.model_prefix + "_pua.model")
    output_pua = tok_pua(input_pua).input_ids

    print(output_extra_id)
    print(output_pua)
    assert output_extra_id == output_pua, "tokenizations do not match"

    # so, this shows that the PUA tokenizer recovers the same token ids as the
    # extra_id tokenizer. This means that a model that already uses the extra_id
    # tokenizer would not need to be retrained. Although this is not sufficient
    # reason by itself to change the tokenizer, it's kind of nice. But are there
    # other reasons to prefer the pua approach?
    # coming next: fast_tokenizers.py

    # 1) Fast tokenizers
    """
    tokfast_extra_id = LlamaTokenizerFast(vocab_file=args.model_prefix + "_extra_id.model")
    output_fast_extra_id = tokfast_extra_id(input_extra_id).input_ids

    tokfast_pua = LlamaTokenizerFast(vocab_file=args.model_prefix + "_pua.model")
    output_fast_pua = tokfast_pua(input_pua).input_ids

    assert output_fast_extra_id == output_fast_pua, "tokenizations do not match"
    """
