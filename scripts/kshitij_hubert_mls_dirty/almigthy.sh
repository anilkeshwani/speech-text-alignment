#!/bin/bash

source /home/kshitij/CoVoST/feat_ext/bin/activate
cd /media/scratch/kshitij/fairseq/examples/hubert/simple_kmeans/
ckpt_path='/media/scratch/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt'
layer=22
nshard=1
rank=0
feat_dir=/mnt/scratch/kshitij/MLS/current_feat_dir/
km_path=/media/scratch/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000
output_kmeans_dir=/mnt/scratch/kshitij/MLS/kmeans_lab_dir/
tsv_dir=/mnt/scratch/sonal/common/mls_splits/

mkdir output_kmeans_dir

echo "Starting the Process."

for split in $tsv_dir*.tsv 
do 
    split=$( basename $split .tsv )
    echo $split >> /mnt/scratch/sonal/common/order_of_tsv.txt
    mkdir $feat_dir
    CUDA_VISIBLE_DEVICES=5 python dump_hubert_feature.py $tsv_dir $split $ckpt_path $layer $nshard $rank $feat_dir   
    CUDA_VISIBLE_DEVICES=5 python dump_km_label.py $feat_dir $split $km_path $nshard $rank $output_kmeans_dir
    rm -rf $feat_dir
done

echo "Process done!"
