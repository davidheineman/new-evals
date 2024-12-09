CLUSTER="\
ai2/jupiter-cirrascale-2,\
ai2/neptune-cirrascale,\
ai2/saturn-cirrascale\
"

# Test configuration
TASK_LIST="\
arc_easy:enlarge::olmes:full \
arc_easy:distractors::olmes:full \
"
# TASK_LIST="\
# arc_easy:rc::olmes:full \
# arc_challenge:rc::olmes:full \
# "
MODEL_LIST="\
weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC \
"

# MODEL_LIST="\
# weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf \
# "

# MODEL_LIST="\
# llama3-70b \
# "

oe-eval \
    --task $TASK_LIST \
    --model $MODEL_LIST \
    --cluster $CLUSTER \
    --model-type hf \
    --beaker-workspace ai2/davidh \
    --beaker-image davidh/oe-eval-metaeval \
    --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
    --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
    --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
    --recompute-metrics \
    --beaker-priority normal