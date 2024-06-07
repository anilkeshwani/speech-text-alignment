# MLS EDA

## Stratified Sample Statistics

Summary and descriptive statistics about the ~25% stratified sample taken of MLS for experiments (_transcripts_stratified_sample_2702009.jsonl_).

### Text

Mean sequence length in tokens: 39.43
Std dev. of sequence length in tokens: 10.32

Sequence length quantiles (tokens):
- 0.001: 14.0
- 0.01: 20.0
- 0.25: 32.0
- 0.5: 39.0
- 0.75: 46.0
- 0.99: 65.0
- 0.999: 74.0

![MLS Stratified Sample: Histogram of text sequence lengths in tokens](/docs/assets/mls_strat_sample_seq_lengths_histogram.png)

### Audio

Mean audio duration (seconds): 14.88
Std dev. of audio duration (seconds): 2.79

Audio duration quantiles:
- 0.001: 10.01
- 0.01: 10.1
- 0.25: 12.51
- 0.5: 14.82
- 0.75: 17.21
- 0.99: 19.88
- 0.999: 20.0

![MLS Stratified Sample: Histogram of audio durations in seconds - MLS](/docs/assets/mls_strat_sample_audio_lengths_histogram.png)

### Speaker Distribution

Mean number of samples by speaker: 492.17
Std dev. of number of samples by speaker: 754.98

Speaker quantiles / samples: 
- 0.001: 1.0
- 0.01: 2.0
- 0.25: 41.0
- 0.5: 135.0
- 0.75: 473.0
- 0.99: 2352.0
- 0.999: 2352.0

The full speaker distribution for the stratified sample is available under [assets/mls_strat_sample_speaker_distribution.json](/docs/assets/mls_strat_sample_speaker_distribution.json). 

![MLS Stratified Sample: Barplot of speaker distribution / samples](/docs/assets/mls_strat_sample_speaker_distribution.png)
