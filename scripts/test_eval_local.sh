OUTPUT_DIR=/root/ai2/metaeval/workspace

# Clear dataloader cache
# rm -rf ~/ai2/.cache/huggingface/datasets/_oe-eval-default_davidh_*
# rm -rf ~/ai2/.cache/huggingface/datasets/dataloader/

# Copy data into repo
DATA_DIR=/root/ai2/metaeval/data # data to copy into Dockerfile
rm -rf ~/ai2/metaeval/olmo-repos/oe-eval-internal/data && \
cp -r $DATA_DIR ~/ai2/metaeval/olmo-repos/oe-eval-internal/data && \

# gsm8k:perturb_rc::olmes \

# gsm_plus::none \
# gsm_symbolic::none \
# gsm_symbolic_p1::none \
# gsm_symbolic_p2::none \
# medmcqa:rc::none \
# medmcqa:mc::none \
# gpqa::none \
# minerva_math_500::none \
# aime::none \

# codex_humaneval:temp0.8 \
# codex_humaneval:temp0.1 \
# paloma_m2d2_s2orc_unsplit::paloma \
# paloma_ptb::paloma \

# autobencher:mc::none \
# autobencher_math::none \
# arxiv_math::llm_compression \

# qwen2-1.5b
# llama3.1-8b \
# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \

oe-eval \
    --task \
        sky_t1::custom_loss \
        numia_math::custom_loss \
        tulu_if::custom_loss \
    --output-dir $OUTPUT_DIR \
    --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/local_testing \
    --model \
        /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
    --model-type hf \
    --run-local \
    --recompute-metrics 
    
# --limit 1000 \


# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC \


# minerva_math_500::tulu \
# aime::tulu \
# medmcqa:rc::none \
# medmcqa:mc::none \
# gsm_plus::none \
# gsm_symbolic::none \

# gpqa:rc::none \
# gpqa:mc::none \
# gpqa:cot::none \

# qwen2.5-14b \


# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
# /oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm \
# /oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646-hf-vllm-2 \


# gsm8k::olmes \
# minerva_math_algebra:perturb_rc::olmes \
# minerva_math_counting_and_probability:perturb_rc::olmes \
# minerva_math_geometry:perturb_rc::olmes \
# minerva_math_intermediate_algebra:perturb_rc::olmes \
# minerva_math_number_theory:perturb_rc::olmes \
# minerva_math_prealgebra:perturb_rc::olmes \
# minerva_math_precalculus:perturb_rc::olmes \

# oe-eval --list-models

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

# coqa:perturb_rc:perturb_rc::olmes \


# drop:perturb_rc:perturb_rc::olmes \
# gsm8k:perturb_rc:perturb_rc::olmes \
# jeopardy:perturb_rc:perturb_rc::olmes \
# naturalqs:perturb_rc:perturb_rc::olmes \
# squad:perturb_rc:perturb_rc::olmes \
# triviaqa:perturb_rc:perturb_rc::olmes \
# mmlu_pro_math:rc::none \
# mmlu_pro_history:rc::none \
# mmlu_pro_math:mc::none \
# mmlu_pro_history:mc::none \


# arc_challenge:enlarge:perturb_rc::olmes \
# arc_easy:enlarge:perturb_rc::olmes \
# boolq:enlarge:perturb_rc::olmes \
# csqa:enlarge:perturb_rc::olmes \
# hellaswag:enlarge:perturb_rc::olmes \
# socialiqa:enlarge:perturb_rc::olmes \
# arc_challenge:distractors:perturb_rc::olmes \
# arc_easy:distractors:perturb_rc::olmes \
# boolq:distractors:perturb_rc::olmes \
# csqa:distractors:perturb_rc::olmes \
# hellaswag:distractors:perturb_rc::olmes \
# socialiqa:distractors:perturb_rc::olmes \
# openbookqa:distractors:perturb_rc::olmes \
# piqa:distractors:perturb_rc::olmes \
# openbookqa:enlarge:perturb_rc::olmes \
# piqa:enlarge:perturb_rc::olmes \