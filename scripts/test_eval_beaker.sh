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
# TASK_LIST="\
# drop:perturb_rc::olmes:full \
# gsm8k:perturb_rc::olmes:full \
# jeopardy:perturb_rc::olmes:full \
# naturalqs:perturb_rc::olmes:full \
# squad:perturb_rc::olmes:full \
# triviaqa:perturb_rc::olmes:full \
# mmlu_pro_math:rc::none \
# mmlu_pro_history:rc::none \
# mmlu_pro_math:mc::none \
# mmlu_pro_history:mc::none \
# "
# TASK_LIST="\
# minerva_math_algebra::tulu \
# minerva_math_intermediate_algebra::tulu \
# agi_eval_lsat-ar:0shot_cot::tulu3 \
# agi_eval_sat-math:0shot_cot::tulu3 \
# mmlu_pro_math:cot::llama3.1 \
# mmlu_pro_psychology:cot::llama3.1 \
# "
# TASK_LIST="\
# agi_eval_lsat-ar:cot::none \
# agi_eval_sat-math:cot::none \
# mmlu_pro_math:cot::none \
# mmlu_pro_psychology:cot::none \
# "
TASK_LIST="\
gsm8k::olmes \
"
# codex_humaneval:temp0.8 \
# mbppplus::ladder \
# mbpp::ladder \
# codex_humanevalplus::ladder \
# MODEL_LIST="\
# weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
# weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC \
# "

# MODEL_LIST="\
# weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm \
# "

MODEL_LIST="\
weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-10xC/step162000-unsharded-hf \
"

# MODEL_LIST="\
# llama3-70b \
# "

# GPUS=4
# MODEL_TYPE=vllm

GPUS=1
MODEL_TYPE=vllm

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