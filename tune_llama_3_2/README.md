# Continued Pre-training & Fine-tuning with torchtune

Code is based on torchtune recipe and config for full fine-tuning of Llama 3.2 1B (and other) models.

- _full_finetune_single_device.py_ is the torchtune "recipe"
- _configs/_ directory contains configuration - this is Hydra style, basically a minimal impl. inspired by Hydra

## Notes

- Used version of Llama 3.2 1B full finetuning recipe (single device) from v0.3.1-rc1-27-g54673b77 (commit [54673b7](https://github.com/pytorch/torchtune/commit/54673b77a3b2d0d8956c42510585c6ea0d979f29))
    - Motivation: The rc version implements an LR scheduler - desired functionality as TinyLlama training runs have been using cosine scheduler.
    - May revert this, or better update it to v0.3.2.
- LICENSE is copied at directory level from torchtune source code - differs from the LICENSE at root taken from fairseq

General Note: torchtune is very dynamic, commits to main every day at the time of writing every few hours. 0 major version -> software in beta
