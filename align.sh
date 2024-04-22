#!/usr/bin/env bash

EMISSION_FIGURE_SAVEPATH="./figures/emissions.png"

rm -fv $EMISSION_FIGURE_SAVEPATH

./align.py \
    --speech-file "./data/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav" \
    --transcript "I had that curiosity beside me at this moment." \
    --emission-figure-savepath $EMISSION_FIGURE_SAVEPATH \
    --segments-output-dir "./segments" \
    --verbose
