OUTPUT_DIR=/root/ai2/metaeval/workspace

# Clear dataloader cache
rm -rf ~/ai2/.cache/huggingface/datasets/_oe-eval-default_davidh_*
rm -rf ~/ai2/.cache/huggingface/datasets/dataloader/

# Copy data into repo
DATA_DIR=/root/ai2/metaeval/data # data to copy into Dockerfile
rm -rf ~/ai2/metaeval/olmo-repos/oe-eval-internal/data && \
cp -r $DATA_DIR ~/ai2/metaeval/olmo-repos/oe-eval-internal/data && \

oe-eval \
    --task \
        mmlu_pro_math:cot::none \
        agi_eval_lsat-lr:cot::none \
    --output-dir $OUTPUT_DIR \
    --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/local_testing \
    --model \
        pythia-160m \
    --model-type hf \
    --run-local \
    --recompute-metrics \
    --limit 100

# oe-eval --list-models

# /oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm \
# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC \
# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf \
# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC \
# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC \
# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-150M-5xC \

# agi_eval_lsat-ar:0shot_cot::tulu3 \ # (::tulu3, :0shot_cot::tulu3) only ::none works on base models
# agi_eval_sat-math:0shot_cot::tulu3 \
# minerva_math_algebra::tulu \ # (::llama3.1, ::tulu) # there's no cot config that works
# minerva_math_intermediate_algebra::tulu \
# mmlu_pro_math:cot::llama3.1 \ # (::llama3.1 and :0shot_cot::tulu3) only ::none works
# mmlu_pro_psychology:cot::llama3.1 \

# coqa:perturb_rc::olmes:full \


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


# arc_challenge:enlarge::olmes:full \
# arc_easy:enlarge::olmes:full \
# boolq:enlarge::olmes:full \
# csqa:enlarge::olmes:full \
# hellaswag:enlarge::olmes:full \
# socialiqa:enlarge::olmes:full \
# arc_challenge:distractors::olmes:full \
# arc_easy:distractors::olmes:full \
# boolq:distractors::olmes:full \
# csqa:distractors::olmes:full \
# hellaswag:distractors::olmes:full \
# socialiqa:distractors::olmes:full \
# openbookqa:distractors::olmes:full \
# piqa:distractors::olmes:full \
# openbookqa:enlarge::olmes:full \
# piqa:enlarge::olmes:full \