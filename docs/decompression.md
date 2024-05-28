# Speeding up Decompression

Attempted calling `tar` passing `-I, --use-compress-program=COMMAND` flag with [pigz](https://github.com/madler/pigz) ([Parallel gzip](https://zlib.net/pigz/)).

```
tar -I pigz -xvf ./mls_english.tar.gz --files-from /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/head_transcripts_stratified_sample_2702009.list
```

## Protocol for installing pigz

- Clone the repository
- Checkout latest release - at time of attempt this was [v2.8](https://github.com/madler/pigz/releases/tag/v2.8)
- Make binaries by calling `make` inside repo at release, per the [README](https://github.com/madler/pigz/blob/fe4894f57739e3039a2ffc2a2a360d35e19bacbe/README)
- (Optional: I moved the binaries to `"${HOME}/bin"` since I add this directory path to my "$PATH" in my ~/.bashrc)
- Test pigz installation via e.g. `pigz --help`
- tar -I pigz -xvf "$compressed_tar" --files-from "$file_list"
