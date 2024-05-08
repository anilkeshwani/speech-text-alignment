import os

root = "/mnt/scratch/kshitij/SPGIspeech/shards/"

files = os.listdir(root)

for i in files:
    os.rename(root+i,root+i[0:-3]+'tsv')