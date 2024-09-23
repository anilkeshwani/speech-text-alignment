import argparse
import json

import jsonlines
import pandas as pd
from evaluate import load
from vllm import LLM, SamplingParams
from whisper_normalizer.basic import BasicTextNormalizer


normalizer = BasicTextNormalizer()

parser = argparse.ArgumentParser(description="Run vLLM for ASR")
parser.add_argument(
    "--token_path",
    type=str,
    help="tokenizer path for the model",
    default="/mnt/data-artemis/duarte/tower_speech/tinyllama_10B_hf_ckpts/tinyllama-1b/4500",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="converted HF model path",
    default="/mnt/data-artemis/duarte/tower_speech/llama2_10B_hf_ckpts_upsample_speech/llama2-7b/3750",
)
parser.add_argument(
    "--data_path",
    type=str,
    help="dataset path for eval",
    default="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/Continual_pretraining__covost_data/CoVoST_complete_test.json",
)

args = parser.parse_args()

tokenizer_path = args.token_path
model_path = args.model_path
data = {"cvst": args.data_path}

# use 128 for LS
sampling_params = SamplingParams(
    stop=["<\s>", "\n", "\\n", "English transcript:", "English speech"], max_tokens=64, temperature=0
)

out_root = "/mnt/scratch-artemis/sonal/MT-experiments/test_outputs/vllm_greedy/"


# prepare into prompt list
def read_data(data_path):
    prompts_df = pd.read_json(data_path)
    ref = prompts_df["output"].to_list()
    prompts = prompts_df["instruction"].to_list()
    return [prompts[i].strip(" ").strip("\n") for i in range(len(prompts))], [
        normalizer(ref[i]) for i in range(len(ref))
    ]


def vllm_generate(out_file, model_path, prompts, ref):
    generated = []
    llm = LLM(model=model_path, tokenizer=tokenizer_path, tokenizer_mode="slow")
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated.append(normalizer(generated_text))

        out_file.write(normalizer(generated_text))
        out_file.write("\n")

    # compute metric
    wer = load("wer")

    results_wer = wer.compute(predictions=generated, references=ref)
    results = {"wer": results_wer}
    print(results)
    return results, generated


for d in data.keys():
    f = open(out_root + "ASR_l7b_IT_{}.txt".format(d), "a")
    prompts, ref = read_data(data[d])
    results, generated = vllm_generate(f, model_path, prompts, ref)

# covost_path="/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/Continual_pretraining__covost_data/CoVoST_complete_test.json"
# LS_clean = "/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-clean.json"
# LS_other = "/mnt/scratch-artemis/kshitij/LLAMA/continued_pretraining/librispeech_eval/after_pretraining/test-other.json"
