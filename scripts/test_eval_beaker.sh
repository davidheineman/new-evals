CLUSTER="\
ai2/jupiter-cirrascale-2,\
ai2/neptune-cirrascale,\
ai2/saturn-cirrascale\
"

# Test configuration
# TASK_LIST="\
# arc_easy:enlarge::olmes:full \
# arc_easy:distractors::olmes:full \
# "
TASK_LIST="\
agi_eval_aqua-rat:perturb_cot::olmes \
agi_eval_gaokao-english:perturb_cot::olmes \
agi_eval_logiqa-en:perturb_cot::olmes \
agi_eval_lsat-ar:perturb_cot::olmes \
agi_eval_lsat-lr:perturb_cot::olmes \
agi_eval_lsat-rc:perturb_cot::olmes \
agi_eval_sat-en-without-passage:perturb_cot::olmes \
agi_eval_sat-en:perturb_cot::olmes \
agi_eval_sat-math:perturb_cot::olmes \
bbh_boolean_expressions:perturb_cot::olmes \
bbh_causal_judgement:perturb_cot::olmes \
bbh_date_understanding:perturb_cot::olmes \
bbh_disambiguation_qa:perturb_cot::olmes \
bbh_dyck_languages:perturb_cot::olmes \
bbh_formal_fallacies:perturb_cot::olmes \
bbh_geometric_shapes:perturb_cot::olmes \
bbh_hyperbaton:perturb_cot::olmes \
bbh_logical_deduction_five_objects:perturb_cot::olmes \
bbh_logical_deduction_seven_objects:perturb_cot::olmes \
bbh_logical_deduction_three_objects:perturb_cot::olmes \
bbh_movie_recommendation:perturb_cot::olmes \
bbh_multistep_arithmetic_two:perturb_cot::olmes \
bbh_navigate:perturb_cot::olmes \
bbh_object_counting:perturb_cot::olmes \
bbh_penguins_in_a_table:perturb_cot::olmes \
bbh_reasoning_about_colored_objects:perturb_cot::olmes \
bbh_ruin_names:perturb_cot::olmes \
bbh_salient_translation_error_detection:perturb_cot::olmes \
bbh_snarks:perturb_cot::olmes \
bbh_sports_understanding:perturb_cot::olmes \
bbh_temporal_sequences:perturb_cot::olmes \
bbh_tracking_shuffled_objects_five_objects:perturb_cot::olmes \
bbh_tracking_shuffled_objects_seven_objects:perturb_cot::olmes \
bbh_tracking_shuffled_objects_three_objects:perturb_cot::olmes \
bbh_web_of_lies:perturb_cot::olmes \
bbh_word_sorting:perturb_cot::olmes \
gsm8k:perturb_cot::olmes \
minerva_math_algebra:perturb_cot::olmes \
minerva_math_counting_and_probability:perturb_cot::olmes \
minerva_math_geometry:perturb_cot::olmes \
minerva_math_intermediate_algebra:perturb_cot::olmes \
minerva_math_number_theory:perturb_cot::olmes \
minerva_math_prealgebra:perturb_cot::olmes \
"
MODEL_LIST="\
weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC \
"

# MODEL_LIST="\
# weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm \
# "

# MODEL_LIST="\
# llama3-70b \
# "

# GPUS=4
# MODEL_TYPE=vllm

GPUS=1
MODEL_TYPE=hf

oe-eval \
    --task $TASK_LIST \
    --model $MODEL_LIST \
    --cluster $CLUSTER \
    --model-type $MODEL_TYPE \
    --gpus $GPUS \
    --beaker-workspace ai2/davidh \
    --beaker-image davidh/oe-eval-metaeval \
    --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
    --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
    --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
    --recompute-metrics \
    --beaker-priority normal