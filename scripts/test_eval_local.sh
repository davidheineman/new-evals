OUTPUT_DIR=/root/ai2/metaeval/workspace

oe-eval \
    --task \
        arc_easy:para::olmes:full \
        arc_easy:enlarge::olmes:full \
        arc_easy:distractors::olmes:full \
    --output-dir $OUTPUT_DIR \
    --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/local_testing \
    --model \
        llama3.2-1b \
    --model-type hf \
    --run-local \
    --recompute-metrics

# oe-eval --list-models

# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-190M-1xC \
# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-190M-2xC \
# /oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-1B-5xC/step81352-unsharded-hf \
# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC \
# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC \
# /oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-150M-5xC \

# arc_easy:rc::olmes:full \
# arc_easy:mc::olmes:full \
# arc_challenge:rc::olmes:full \
# arc_challenge:mc::olmes:full \
# piqa:rc::olmes:full \
# piqa:mc::olmes:full \
# boolq:rc::olmes \
# boolq:mc::olmes \
# hellaswag:rc::olmes \
# hellaswag:mc::olmes \
# codex_humaneval \
# gsm8k \
