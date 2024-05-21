# MLS Data Preparation

Given the weight of the MLS dataset, we want to unpack a targeted subset of it for training. We will do this in the following steps, focussing on the train set since the dev and test splits are a manageable size:

1. Convert the transcripts.txt file to JSON lines format - this is useful downstream when we will concatenate several datasets including GigaSpeech, VoxPopuli, SPGISpeech and others, as well as text-only datasets. 
2. Transcribe all the transcripts into "universal roman" characters via _uroman_
3. Select a stratified sample of ~25% of the train set, stratifying by speaker to obtain an maximally diverse subset (e.g. vocal characteristics, accent)
4. Create a manifest (file list) of the files to extract given this stratified sample
5. Unpack the files given this manifest
6. Compute alignments for all audios and text transcripts with the MMS aligner and save token-level audio segments to disk with accompanying transcripts
    - a JSON lines file containing the ID, transcript, text token, start time, end time and duration (end time - start time + 2*context padding)
7. Featurize the token-wise segmented audio files with a HuBERT model
8. Fit or re-use a k-means model to "transcribe" the segmented audio clips into DSUs, or HuBERT speech tokens
    - Add the DSUs as a field to the JSON lines
    - Choice of k-means model. The k-means model can be fitted on:
        - the HuBERT features of the whole audios - in which case HuBERT featurization of the whole audios is required
        - the generated HuBERT features
9. Interleave the data:
    - need to create contiguous _spans_ of speech or text tokens - flipping a coin each time would not generate uninterrupted sequences of the same modality of tokens. Might be best to use a simple geometric distribution where the probability of staying with the same modality decays to $0$ as the sequence continues
        - how to tune this based on the distribution of sequence lengths to result in approx. 50% of text tokens being transcribed?
        - I think it's desirable to think about the ratio transcribed in terms of the proportion of text tokens that are converted to speech tokens
    - for each (audio) ID, compute these spans - can be done simply with the lengths of the original transcripts in terms of numbers of tokens
    - insert additional "modality" tokens bookending the spans of text or speech tokens
    - write a function that takes as input a single sample, segmented into token-wise subsamples (of audio and text tokens) and returns an interleaved sample, with joined spans including the modality (switch) tokens at switch points
