#!/bin/bash
answersfile="/absolute_path_to/playground/data/eval/MME/answers/mrt6_4"
templatefile="/absolute_path_to/playground/data/eval/MME/eval_tool/Your_Results"
caloutfile="/absolute_path_to/playground/data/eval/MME/mrt6_4"
#
python ./MRT/eval/mme_mrt.py \
    --model-path "/path_to/llava-pretrain-vicuna-7b-v1.3" \
    --model-base "/path_to/vicuna-7b-v1.3" \
    --question-file /path_to/playground/data/eval/MME/llava_mme_yingNew.jsonl \
    --image-folder /path_to/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file $answersfile\
    --temperature 0.4 \
    --max_new_tokens 8 \
    --conv-mode vicuna_v1 \
    --result_dir ./mrt-6-4

cd ./playground/data/eval/MME
#
python eval_mme_ying.py --path $answersfile --output-path $caloutfile --template-path $templatefile
#
cd eval_tool
#
python calculation.py --results_dir $caloutfile