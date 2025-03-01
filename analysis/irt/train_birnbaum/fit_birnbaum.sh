#!/bin/bash

for task in arc_challenge arc_challenge_mc arc_easy arc_easy_mc boolq boolq_mc csqa csqa_mc gsm8k hellaswag_mc mbpp mbppplus minerva mmlu mmlu_mc openbookqa openbookqa_mc piqa piqa_mc socialiqa socialiqa_mc winogrande winogrande_mc
do
  python fit_birnbaum.py --input_data "tasks/${task}"
done
