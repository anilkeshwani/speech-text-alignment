# TODO

- [x] Fix CTC decoding issue for aberrant sample in VoxPopuli
- [ ] Create interleaved dataset with VoxPopuli
- [ ] Write converter script for old datasets containing alignments and DSUs (MLS)
    - [ ] -> run this converter script on MLS
    - [ ] -> run this converter script on GigaSpeech
- [ ] As a check: Create an interleaved dataset with the MLS aligned speech-text data that you converted


- [x] Check where TEXT_KEY_DEFAULT is being used and if it can be uniformly removed

Easy:
- [x] Rename script to uromanize.py
- [x] move segment_tokens.py and align_and_segment.py - as well as possibly others - to ./scripts/auxiliary/ subdirectory or similar
- [x] Rename generate_interleaved_data.py to interleave_data.py
- [x] finally: Ensure the pre-commit works and formats - and in particular isorts - everything that has been changed


- later: make hubert encoding optional in align_and_hubert_encode.py
- find a better name for align_and_hubert_encode.py

# Improvements

- Support for k-means clustering via [FAISS clustering](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization) - Motivation: speed. Not a priority if the k-means clustering is a bottlenecked due to computational speed. See also [_Implementing K-Means clustering with FAISS_](https://www.kdnuggets.com/2021/01/k-means-faster-lower-error-scikit-learn.html) and [FAISS k-means snippet](/snippets/faiss_kmeans.py).
- Can we read directly from a compressed zip file with `fairseq.data.audio.audio_utils.get_features_or_waveform` since this allows specification of the `path` argument as either a `File path in the format of "<.npy/.wav/.flac path>" or "<zip path>:<byte offset>:<byte length>"` according to the docstring. Internally, there is a call to `get_features_or_waveform_from_stored_zip` which needs a byte offset in the zip file.
    - Do MLS, GigaSpeech or other large datasets that come as compressed zips provide these offsets? Can we compute them? Is this faster that decompressing _a priori_?


# Done

- [x] Check that scripts/dump_hubert_features.py still returns _identical_ outputs to the original fairseq script
    - IIRC when I checked this, there were very nearly identical outputs to those returned by the original fairseq scripts - differences were small floating point errors (albeit larger than the error of machine precision) and the HuBERT DSUs that were output by the model + k-means matched across tests where I ran the HuBERT and other code 👍
- [ ] For HuBERT feature generation, perform a check that HuBERT features are identical when taking in a (segmented) audio written to disk via sox.Transformer.build_file() as compared to the waveforms that I use which are read from torchaudio.load
    - Reading the source code of HuBERT shows the feature dumper uses `fairseq.data.audio.audio_utils.get_waveform` to obtain normalized waveforms given the default arguments c.f. the MMS alignment code which uses the `load` function returned by `torchaudio._backend.utils.get_load_func`
    - not sure I checked this via the method described above here but I think the audio waveforms were reasonable when I performed _ad hoc_ checks, mainly of the code to ensure that the audio was not being left unnormalized or being normalized twice. Also the HuBERT DSUs returned matched the fairseq implementation, which implies that the audio is fine or at least usable for the purposes we need.
