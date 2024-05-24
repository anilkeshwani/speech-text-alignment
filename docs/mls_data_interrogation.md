# MLS Data Interrogation

The [Multilingual LibriSpeech (MLS) dataset](https://arxiv.org/pdf/2012.03411) can be [downloaded from OpenSLR](https://www.openslr.org/94/) in subsets by language. 

The English subset is available as the original _flac_ audio files (2.4TB) or compressed _opus_ files (651GB). The GNU zipped tape-archived original flacs can be downloaded (in ~30 hours at ~25MB/s) with `wget` via:

```bash
wget https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz # takes ~30 hours at ~25MB/s download speed
```

Since the archive is 2.4TB, a manifest of files contained can be obtained via the following command, which takes ~5:10 to complete:

```bash
tar --list -f mls_english.tar.gz > archive_contents.txt # run inside a tmux session; takes 5+ hours to complete
```

Audio files (flacs) in MLS are organised into subdirectories for 
1. language
2. split
3. (audio)
4. speaker ID
5. book ID 

in the above order. 

Audio filenames are of the format `f"{speaker_id}_{book_id}_audio_id.flac"`.

For example:

```
mls_english/dev/audio/1982/1551/1982_1551_000037.flac
```

This results in a file containing 10,855,734 lines - this includes entries for directories. 

```bash
wc archive_contents.txt #  10855734  10855734 618098858 archive_contents.txt
```

We can filter for files, which should reduce the file to 10,815,627 lines.

```bash
grep "\." archive_contents.txt > archive_contents_files.txt
wc archive_contents_files.txt # 10815627  10815627 616697973 archive_contents_files.txt
```

Subsequently count the number of files of each type to validate the data:

```bash
./scripts/mls/count_file_types.py "/mnt/scratch-artemis/anilkeshwani/data/MLS/archive_contents_files.txt"
```

```
{
    'flac': 10815613, 
    'txt': 14
}
```

We can separate out the audio (flac) files:

```bash
grep "\.flac" archive_contents.txt > archive_contents_flac_files.txt
wc archive_contents_flac_files.txt # 10815613  10815613 616697369 archive_contents_flac_files.txt
```

We can separate out the text files, which contain metadata and file manifests and unpack these:

```bash
grep "\.txt" archive_contents.txt > archive_contents_text_files.txt
wc archive_contents_text_files.txt
tar -xvf mls_english.tar.gz --files-from archive_contents_text_files.txt
```

MLS contains the following text files:
- mls_english/metainfo.txt
- mls_english/dev/transcripts.txt
- mls_english/dev/segments.txt
- mls_english/test/segments.txt
- mls_english/test/transcripts.txt
- mls_english/train/transcripts.txt
- mls_english/train/segments.txt
- mls_english/train/limited_supervision/1hr/0/handles.txt
- mls_english/train/limited_supervision/1hr/1/handles.txt
- mls_english/train/limited_supervision/1hr/2/handles.txt
- mls_english/train/limited_supervision/1hr/3/handles.txt
- mls_english/train/limited_supervision/1hr/4/handles.txt
- mls_english/train/limited_supervision/1hr/5/handles.txt
- mls_english/train/limited_supervision/9hr/handles.txt

The metainfo.txt file shows metadata about the whole dataset and specifically the following fields:

- speaker
- gender
- partition
- minutes
- book id
- title
- chapter

```
 SPEAKER   |   GENDER   | PARTITION  |  MINUTES   |  BOOK ID   |             TITLE              |            CHAPTER            
  10232    |     M      |   secret   |   17.148   |   10057    | Expression of the Emotions in Man and Animals | Ch. II: General Principles of Expression, continued
   9508    |     F      |   secret   |   9.347    |   10105    | Stephen: A Soldier of the Cross | Good Tidings Out of the Desert
   9508    |     F      |   secret   |   8.123    |   12959    |         Vanished Hand          | CHAPTER II - WHAT WAS WRITTEN 
  10375    |     M      |   secret   |   10.803   |   10173    | Dutch Fairy Tales for Young Folks |   SANTA KLAAS AND BLACK PETE  
  10375    |     M      |   secret   |   6.764    |   10244    | Grimm's Fairy Tales - Retold in One-Syllable Words |          Hans in Luck         
  10655    |     M      |   secret   |   17.841   |   10173    | Dutch Fairy Tales for Young Folks | THE FARM THAT RAN AWAY AND CAME BACK
  10454    |     M      |   secret   |   1.782    |   10203    |             Verses             | The Cradle Tomb in Westminster Abbey
  10454    |     M      |   secret   |   2.316    |   10203    |             Verses             |          Commissioned         
  10454    |     M      |   secret   |   2.362    |   10335    | Grand'ther Baldwin's Thanksgiving, with Other Ballads and Poems |         Friar Anselmo         
```

MLS dataset splits ({train, dev, test}) are split into subdirectories, each containing transcripts.txt and segments.txt files. 

Each transcripts.txt contains:
- file identifier
- transcript

```
4800_10003_000000       oh my dear you must see him he expects you she answered almost gayly the procession of three moved down the long room towards a door phyllis's hand guiding the wheel-chair
4800_10003_000001       it was quite as much fun well almost as much hearing her as it would have been to play all of the contented and otherwise elderly people who inhabited the boarding-house with phyllis
4800_10003_000002       the man stole out and shut the door softly phyllis herself rose and went toward the window and busied herself in braiding up her hair there was almost silence in the room for a few minutes
4800_10003_000003       has it said phyllis it was like mrs harrington that careful planning of even where she should be put is mr harrington in his day-room now
4800_10003_000004       and she insisted that the pink paper stay on the electric lights after about a week of this phyllis suddenly remembered that she had not been selfish at all yet
4800_10003_000005       surprise i-i'm glad you like it said his wife shyly still backing away of course he'd like it said mrs de guenther's kind staccato voice behind him kiss your husband and tell him he's welcome home phyllis child
4800_10003_000006       you have everything that could be asked even to a certain cheerfulness of outlook which poor angela naturally lacks in a measure but-but what about me asked phyllis braithwaite a little piteously in answer to all this
4800_10003_000007       i've bought myself lots of things she defended herself most of this is really for me and-i can't help being good to him it's only common humanity
4800_10003_000008       his little crumpled black muzzle on the pillow close to allan's contented sleeping face she felt as if she wanted to cry the pathetic lack of interests which made the coming of a new little dog such an event
4800_10003_000009       she wondered afterwards how she could have spoken with that hard serenity how she could have gone steadily on with story after story poem after poem till allan's grip on her hands relaxed and he fell into a heavy tired sleep
```

Each segments.txt contains:

- file identifier
- URL to access file (as mp3)
- timestamp for segment start in seconds
- timestamp for segment end in seconds

```
4800_10003_000000       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_05_widdemer_64kb.mp3      401.76  417.57
4800_10003_000001       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_03_widdemer_64kb.mp3      238.58  252.82
4800_10003_000002       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_07_widdemer_64kb.mp3      1160.28 1174.41
4800_10003_000003       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_07_widdemer_64kb.mp3      599.02  612.41
4800_10003_000004       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_08_widdemer_64kb.mp3      363.34  376.76
4800_10003_000005       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_10_widdemer_64kb.mp3      993.58  1013.33
4800_10003_000006       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_04_widdemer_64kb.mp3      224.67  243.66
4800_10003_000007       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_09_widdemer_64kb.mp3      568.02  580.43
4800_10003_000008       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_12_widdemer_64kb.mp3      269.04  285.39
4800_10003_000009       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_14_widdemer_64kb.mp3      240.72  258.77
```

We can check that the number of audio files matches the total number across transcripts.txt files from the train, dev and test splits:

```bash
find . -name "transcripts.txt" | xargs -I{} wc -l {}
```

```
3807 ./mls_english/dev/transcripts.txt
10808037 ./mls_english/train/transcripts.txt
3769 ./mls_english/test/transcripts.txt
```

This is a total of 10,815,613 flac files which matches the line count above (`wc archive_contents_flac_files.txt`).
