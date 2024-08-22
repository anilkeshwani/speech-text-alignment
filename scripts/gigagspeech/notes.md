# Notes on GigaSpeech inc. Preprocessing

I ran preprocessing twice - once when the script loaded the whole preprocessed JSON lines manifest file into memory and then dumped this onto disk in one shot and another which periodically writes to an open file handle.

I hashed both files to check they were identical: they were not. **The difference is accounted for in capitalisation differences produced after running truecasing using the [truecase](https://github.com/daltonfury42/truecase) Python package.**. See examples below.


### Hashes do not match

```bash
(main) anilkeshwani@poseidon:/mnt/scratch-artemis/anilkeshwani/data/GigaSpeech_HF/sorted$ openssl dgst -sha256 GigaSpeech.jsonl
SHA2-256(GigaSpeech.jsonl)= d77950ffa2bc9d8210bd192772e8ffd96f2957e1c8c015da9b8d677e4655c417
(main) anilkeshwani@poseidon:/mnt/scratch-artemis/anilkeshwani/data/GigaSpeech_HF/sorted$ openssl dgst -sha256 GigaSpeech_streamed.jsonl
SHA2-256(GigaSpeech_streamed.jsonl)= 1ee713453c5cab970063f7260595cd9b62a699aa8ace68ad2d590bc78fc6f857
```

### Differences

Take the top 2,000 lines of each JSON lines manifest and run diff:

```bash
(main) anilkeshwani@poseidon:/mnt/scratch-artemis/anilkeshwani/data/GigaSpeech_HF/sorted$ diff -s ./*top*
```

In the first pair, for example, "thrice" is capitalised in the second but not the first.

> Three civil Brawls, bred of an airy word by thee, old Capulet, and Montague, have **thrice** disturb'd the quiet of our streets and made Verona's ancient citizens cast by their grave Beseeming ornaments to wield old partisans, in hands as old, Cank'Red with peace,

> Three civil Brawls, bred of an airy word by thee, old Capulet, and Montague, have **Thrice** disturb'd the quiet of our streets and made Verona's ancient citizens cast by their grave Beseeming ornaments to wield old partisans, in hands as old, Cank'Red with peace,

In the second example, the difference is the other way around, where the first is capitalised and the second is not:
> maiden Blush Bepaint

> maiden blush Bepaint


```json
{
    "segment_id": "AUD0000000007_S0000120",
    "text": "THREE CIVIL BRAWLS <COMMA> BRED OF AN AIRY WORD BY THEE <COMMA> OLD CAPULET <COMMA> AND MONTAGUE <COMMA> HAVE THRICE DISTURB'D THE QUIET OF OUR STREETS AND MADE VERONA'S ANCIENT CITIZENS CAST BY THEIR GRAVE BESEEMING ORNAMENTS TO WIELD OLD PARTISANS <COMMA> IN HANDS AS OLD <COMMA> CANK'RED WITH PEACE <COMMA>",
    "text_processed": "Three civil Brawls, bred of an airy word by thee, old Capulet, and Montague, have thrice disturb'd the quiet of our streets and made Verona's ancient citizens cast by their grave Beseeming ornaments to wield old partisans, in hands as old, Cank'Red with peace,",
    "audio_id": "AUD0000000007",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/90f594a6b0416f449f51aac169b963190c1258e647a477a4a796e09bd14e2b7b/s_chunks_0001/AUD0000000007_S0000120.wav",
    "speaker": "N/A",
    "begin_time": 473.30999755859375,
    "end_time": 491.44000244140625,
    "title": "Romeo and Juliet",
    "url": "http//www.archive.org/download/romeo_and_juliet_librivox/romeo_and_juliet_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000007.opus"
}
{
    "segment_id": "AUD0000000007_S0000120",
    "text": "THREE CIVIL BRAWLS <COMMA> BRED OF AN AIRY WORD BY THEE <COMMA> OLD CAPULET <COMMA> AND MONTAGUE <COMMA> HAVE THRICE DISTURB'D THE QUIET OF OUR STREETS AND MADE VERONA'S ANCIENT CITIZENS CAST BY THEIR GRAVE BESEEMING ORNAMENTS TO WIELD OLD PARTISANS <COMMA> IN HANDS AS OLD <COMMA> CANK'RED WITH PEACE <COMMA>",
    "text_processed": "Three civil Brawls, bred of an airy word by thee, old Capulet, and Montague, have Thrice disturb'd the quiet of our streets and made Verona's ancient citizens cast by their grave Beseeming ornaments to wield old partisans, in hands as old, Cank'Red with peace,",
    "audio_id": "AUD0000000007",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/90f594a6b0416f449f51aac169b963190c1258e647a477a4a796e09bd14e2b7b/s_chunks_0001/AUD0000000007_S0000120.wav",
    "speaker": "N/A",
    "begin_time": 473.30999755859375,
    "end_time": 491.44000244140625,
    "title": "Romeo and Juliet",
    "url": "http//www.archive.org/download/romeo_and_juliet_librivox/romeo_and_juliet_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000007.opus"
}
```

```json
{
    "segment_id": "AUD0000000007_S0000642",
    "text": "THOU KNOWEST THE MASK OF NIGHT IS ON MY FACE ELSE WOULD A MAIDEN BLUSH BEPAINT MY CHEEK FOR THAT WHICH THOU HAST HEARD ME SPEAK TO-NIGHT <PERIOD>",
    "text_processed": "Thou Knowest the mask of night is on my face else would a maiden Blush Bepaint my cheek for that which thou hast heard me speak To-Night.",
    "audio_id": "AUD0000000007",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/44a59079882b91a916bd72b70ec9e40120c08bef975754bddbbd397824e612a2/s_chunks_0000/AUD0000000007_S0000642.wav",
    "speaker": "N/A",
    "begin_time": 3279.31005859375,
    "end_time": 3287.5400390625,
    "title": "Romeo and Juliet",
    "url": "http//www.archive.org/download/romeo_and_juliet_librivox/romeo_and_juliet_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000007.opus"
}
{
    "segment_id": "AUD0000000007_S0000642",
    "text": "THOU KNOWEST THE MASK OF NIGHT IS ON MY FACE ELSE WOULD A MAIDEN BLUSH BEPAINT MY CHEEK FOR THAT WHICH THOU HAST HEARD ME SPEAK TO-NIGHT <PERIOD>",
    "text_processed": "Thou Knowest the mask of night is on my face else would a maiden blush Bepaint my cheek for that which thou hast heard me speak To-Night.",
    "audio_id": "AUD0000000007",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/44a59079882b91a916bd72b70ec9e40120c08bef975754bddbbd397824e612a2/s_chunks_0000/AUD0000000007_S0000642.wav",
    "speaker": "N/A",
    "begin_time": 3279.31005859375,
    "end_time": 3287.5400390625,
    "title": "Romeo and Juliet",
    "url": "http//www.archive.org/download/romeo_and_juliet_librivox/romeo_and_juliet_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000007.opus"
}
```

```json
{
    "segment_id": "AUD0000000012_S0000200",
    "text": "WHO COM MANDED AN ARMY IN BRABANT <COMMA> AT THE OPENING OF THE FIRST CAMPAIGN AGAINST THE FRENCH JACOBINS <PERIOD>",
    "text_processed": "Who com Manded an Army in Brabant, at the opening of the first campaign against the French Jacobins.",
    "audio_id": "AUD0000000012",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/442bafa43804d47a1af36ad7a58dc28f1ccd1f6e1c14dc2c026cdd2d57e2b35e/s_chunks_0003/AUD0000000012_S0000200.wav",
    "speaker": "N/A",
    "begin_time": 1376.1600341796875,
    "end_time": 1383.4300537109375,
    "title": "Memoir on the Life and Character of the Rev. Prince Demetrius A. de Gallitzin",
    "url": "http//www.archive.org/download/princegallitzin_dv_librivox/princegallitzin_dv_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000012.opus"
}
{
    "segment_id": "AUD0000000012_S0000200",
    "text": "WHO COM MANDED AN ARMY IN BRABANT <COMMA> AT THE OPENING OF THE FIRST CAMPAIGN AGAINST THE FRENCH JACOBINS <PERIOD>",
    "text_processed": "Who COM Manded an Army in Brabant, at the opening of the first campaign against the French Jacobins.",
    "audio_id": "AUD0000000012",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/442bafa43804d47a1af36ad7a58dc28f1ccd1f6e1c14dc2c026cdd2d57e2b35e/s_chunks_0003/AUD0000000012_S0000200.wav",
    "speaker": "N/A",
    "begin_time": 1376.1600341796875,
    "end_time": 1383.4300537109375,
    "title": "Memoir on the Life and Character of the Rev. Prince Demetrius A. de Gallitzin",
    "url": "http//www.archive.org/download/princegallitzin_dv_librivox/princegallitzin_dv_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000012.opus"
}
```

```json
{
    "segment_id": "AUD0000000012_S0000576",
    "text": "HIS LITTLE LONE CABIN <COMMA>",
    "text_processed": "His little lone cabin,",
    "audio_id": "AUD0000000012",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/34bcd9f587fb6d49707c11336ff956953bdbf293bc22eabeb1a7668c890c1042/l_chunks_0053/AUD0000000012_S0000576.wav",
    "speaker": "N/A",
    "begin_time": 3553.320068359375,
    "end_time": 3555.050048828125,
    "title": "Memoir on the Life and Character of the Rev. Prince Demetrius A. de Gallitzin",
    "url": "http//www.archive.org/download/princegallitzin_dv_librivox/princegallitzin_dv_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000012.opus"
}
{
    "segment_id": "AUD0000000012_S0000576",
    "text": "HIS LITTLE LONE CABIN <COMMA>",
    "text_processed": "His little Lone cabin,",
    "audio_id": "AUD0000000012",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/34bcd9f587fb6d49707c11336ff956953bdbf293bc22eabeb1a7668c890c1042/l_chunks_0053/AUD0000000012_S0000576.wav",
    "speaker": "N/A",
    "begin_time": 3553.320068359375,
    "end_time": 3555.050048828125,
    "title": "Memoir on the Life and Character of the Rev. Prince Demetrius A. de Gallitzin",
    "url": "http//www.archive.org/download/princegallitzin_dv_librivox/princegallitzin_dv_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000012.opus"
}
```

```json
{
    "segment_id": "AUD0000000012_S0000724",
    "text": "WHEN THE SAME UNJUST STEWARD <COMMA> WHOM HE HAD APPOINTED OVER ALL HIS GOODS <COMMA>",
    "text_processed": "When the same unjust steward, whom he had appointed over all his goods,",
    "audio_id": "AUD0000000012",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/e6bec2604d8cd88f2d4214507548646831c61d1958f019bdfe4a3011b0c5ea95/s_chunks_0016/AUD0000000012_S0000724.wav",
    "speaker": "N/A",
    "begin_time": 4365.47998046875,
    "end_time": 4370.31005859375,
    "title": "Memoir on the Life and Character of the Rev. Prince Demetrius A. de Gallitzin",
    "url": "http//www.archive.org/download/princegallitzin_dv_librivox/princegallitzin_dv_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000012.opus"
}
{
    "segment_id": "AUD0000000012_S0000724",
    "text": "WHEN THE SAME UNJUST STEWARD <COMMA> WHOM HE HAD APPOINTED OVER ALL HIS GOODS <COMMA>",
    "text_processed": "When the same unjust Steward, whom he had appointed over all his goods,",
    "audio_id": "AUD0000000012",
    "path": "/home/anilkeshwani/.cache/huggingface/datasets/downloads/extracted/e6bec2604d8cd88f2d4214507548646831c61d1958f019bdfe4a3011b0c5ea95/s_chunks_0016/AUD0000000012_S0000724.wav",
    "speaker": "N/A",
    "begin_time": 4365.47998046875,
    "end_time": 4370.31005859375,
    "title": "Memoir on the Life and Character of the Rev. Prince Demetrius A. de Gallitzin",
    "url": "http//www.archive.org/download/princegallitzin_dv_librivox/princegallitzin_dv_librivox_64kb_mp3.zip",
    "source": 0,
    "category": 28,
    "original_full_path": "audio/audiobook/P0001/AUD0000000012.opus"
}
```
