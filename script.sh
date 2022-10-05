#!/bin/bash

while getopts c:r:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dataset_dir=${OPTARG};;
    esac
done

for i in $(seq 1 $reps); do
    python scripts/train.py --train-path $dataset_dir/English/E-c/2018-E-c-En-train.txt --dev-path $dataset_dir/English/E-c/2018-E-c-En-dev.txt --seed=$i --device cuda:$cuda --filename english
    python scripts/test.py --test-path $dataset_dir/English/E-c/2018-E-c-En-test-gold.txt --model-path english_checkpoint.pt --device cuda:$cuda
done

for i in $(seq 1 $reps); do
    python scripts/train.py --train-path $dataset_dir/Spanish/E-c/2018-E-c-Es-train.txt --dev-path $dataset_dir/Spanish/E-c/2018-E-c-Es-dev.txt --seed=$i --lang Spanish --device cuda:$cuda --filename spanish
    python scripts/test.py --test-path $dataset_dir/Spanish/E-c/2018-E-c-Es-test-gold.txt --model-path spanish_checkpoint.pt --lang Spanish --device cuda:$cuda
done


for i in $(seq 1 $reps); do
    python scripts/train.py --train-path $dataset_dir/Arabic/E-c/2018-E-c-Ar-train.txt --dev-path $dataset_dir/Arabic/E-c/2018-E-c-Ar-dev.txt --seed=$i --lang Arabic --device cuda:$cuda --filename arabic
    python scripts/test.py --test-path $dataset_dir/Arabic/E-c/2018-E-c-Ar-test-gold.txt --model-path arabic_checkpoint.pt --lang Arabic --device cuda:$cuda
done