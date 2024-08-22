# To Do

- [ ] Check that scripts/dump_hubert_features.py still returns _identical_ outputs to the original fairseq script
- [ ] For HuBERT feature generation, perform a check that HuBERT features are identical when taking in a (segmented) audio written to disk via sox.Transformer.build_file() as compared to the waveforms that I use which are read from torchaudio.load
    - Reading the source code of HuBERT shows the feature dumper uses `fairseq.data.audio.audio_utils.get_waveform` to obtain normalized waveforms given the default arguments c.f. the MMS alignment code which uses the `load` function returned by `torchaudio._backend.utils.get_load_func`

## Improvements

- Support for k-means clustering via [FAISS clustering](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization) - Motivation: speed. Not a priority if the k-means clustering is a bottlenecked due to computational speed. See also [_Implementing K-Means clustering with FAISS_](https://www.kdnuggets.com/2021/01/k-means-faster-lower-error-scikit-learn.html) and [FAISS k-means snippet](/snippets/faiss_kmeans.py).
- Can we read directly from a compressed zip file with `fairseq.data.audio.audio_utils.get_features_or_waveform` since this allows specification of the `path` argument as either a `File path in the format of "<.npy/.wav/.flac path>" or "<zip path>:<byte offset>:<byte length>"` according to the docstring. Internally, there is a call to `get_features_or_waveform_from_stored_zip` which needs a byte offset in the zip file.
    - Do MLS, GigaSpeech or other large datasets that come as compressed zips provide these offsets? Can we compute them? Is this faster that decompressing _a priori_?
