import math
import os

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


output_root = "/mnt/scratch/sonal/common/mls_splits/"
audio_files_dir = "/mnt/scratch/sonal/common/moved_wav_files/"
audio_list = os.listdir(audio_files_dir)

number_audio_files = len(audio_list)

number_of_audio_files_per_folder = 100000
number_of_sub_dirs = math.floor(number_audio_files / number_of_audio_files_per_folder)
print(f"Total number of splits: {number_of_sub_dirs}")

dist = np.array_split(audio_list, number_of_sub_dirs)
dist = [list(i) for i in dist]


## ADD ABOUT GETTING THE NUMBER OF SAMPLES IN THE AUDIO FILE
for n, i in tqdm(enumerate(dist)):
    file_name = output_root + f"split_{n}" + ".tsv"
    with open(file_name, "w+") as f:
        f.write(audio_files_dir + "\n")
        for j in i:
            _, nparr = wavfile.read(audio_files_dir + j)
            num = str(len(nparr))
            towrite = j + "\t" + num + "\n"
            f.write(towrite)
