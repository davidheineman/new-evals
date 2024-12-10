import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_INTERMEDIATE, MODEL_LIST_MIXES, OE_EVAL_OFFICIAL_MODELS
from analysis.utils.constants_models import RC_TASKS_OLMES, PARA_TASKS_OLMES
from analysis.utils.constants_models import WEKA_CLUSTERS

MODEL_LIST_ALL = MODEL_LADDER_LIST + MODEL_LIST_INTERMEDIATE + MODEL_LIST_MIXES
MODEL_LIST_ALL += OE_EVAL_OFFICIAL_MODELS

TASK_LIST_ALL = RC_TASKS_OLMES + PARA_TASKS_OLMES
SYNTHETIC_TASKS = [
    "arc_easy:enlarge::olmes:full",
    "arc_challenge:enlarge::olmes:full",
    "arc_easy:distractors::olmes:full",
    "arc_challenge:distractors::olmes:full",
]
TASK_LIST_ALL += SYNTHETIC_TASKS
TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' not in task] # exclude mmlu (long arg lists may crash beaker! https://github.com/allenai/beaker/issues/5530)

# # FOR TESTING
# MODEL_LIST_ALL = [MODEL_LIST_ALL[0]] # <- only use first model!
# TASK_LIST_ALL = SYNTHETIC_TASKS # <- only use synthetic tasks!
# # MODEL_LIST_ALL = OE_EVAL_OFFICIAL_MODELS
# MODEL_LIST_ALL = [model for model in MODEL_LIST_INTERMEDIATE if 'step58000-unsharded-hf' in model]

# Models that broke on 1 GPU HF
# MODEL_LIST_ALL = [
#     "dclm-1b", "dclm-7b", "gemma-7b", "gemma2-9b", "llama2-7b",
#     "llama3-70b", "llama3.1-70b", "olmo-7b-0424", "qwen2.5-32b", "qwen2.5-72b",
#     "llama-7b",
#     "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm",
# ]
# MODEL_LIST_ALL = ['llama3-70b']

# TODO: In practice, I'd want to eval peteish7, 13 on vLLM. Add this to the script so I can launch jobs easily

GPUS = 1
MODEL_TYPE = 'hf'

# From my testing, looks like anything less than 4 GPUs on 13B+ models (or Gemma 7B+) breaks
# Also 70B model do not work on neptune (L40s)
# GPUS = 4 
# MODEL_TYPE = 'vllm'


def run_eval(model_list, task_list):
    if isinstance(task_list, list): 
        task_list = ' '.join(task_list)

    command = f"""
    oe-eval \
        --model {model_list} \
        --task {task_list} \
        --cluster {WEKA_CLUSTERS} \
        --model-type {MODEL_TYPE} \
        --gpus {GPUS} \
        --beaker-workspace ai2/davidh \
        --beaker-image davidh/oe-eval-metaeval \
        --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
        --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
        --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
        --recompute-metrics \
        --beaker-priority normal
    """
    command = command.replace('  ', '') # remove extra spacing!

    print(f'Executing command:\n{command}')
    os.system(command)


def main():
    print(f'Launching {len(MODEL_LIST_ALL)} models on {len(TASK_LIST_ALL)} tasks (10 second sleep to confirm...)')
    time.sleep(10)

    for model in MODEL_LIST_ALL:
        run_eval(model, TASK_LIST_ALL)


if __name__ == '__main__': main()


# Want more models?
# https://github.com/mlfoundations/open_lm
# open_lm_11m
# open_lm_25m
# open_lm_87m
# open_lm_160m
# open_lm_411m
# open_lm_830m
# open_lm_1b
# open_lm_3b
# open_lm_7b

# Scaling data constrained LLMs (https://arxiv.org/abs/2305.16264)
# https://huggingface.co/datablations#models / https://github.com/huggingface/datablations

# (probably your best bet) 20M to 3.3B:
# https://huggingface.co/KempnerInstituteAI/loss-to-loss
