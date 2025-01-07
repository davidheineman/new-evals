import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_INTERMEDIATE, MODEL_LIST_MIXES, OE_EVAL_BASE_MODELS, OE_EVAL_INSTRUCT_MODELS, OE_EVAL_ALL_MODELS
from analysis.utils.constants_final_6_ckpts import MODEL_LIST_FINAL_SIX_CKPTS
from analysis.utils.constants_models import WEKA_CLUSTERS
from analysis.utils.constants_tasks import MC_TASKS_COPY_COLORS

# OLMES Classic Tasks
from analysis.utils.constants_tasks import RC_TASKS_OLMES, PARA_TASKS_OLMES, ENLARGE_TASKS_OLMES, DISTRACTORS_TASKS_OLMES

# OLMES Gen Tasks
from analysis.utils.constants_tasks import GEN_TASKS_OLMES, GEN_TASKS_OLMES_PERTURB_RC

# CoT tasks (mainly Tulu tasks)
from analysis.utils.constants_tasks import AGI_EVAL_MC, AGI_EVAL_COT
from analysis.utils.constants_tasks import MMLU_PRO_MC, MMLU_PRO_RC, MMLU_PRO_COT
from analysis.utils.constants_tasks import MINERVA_MC, MINERVA_COT
from analysis.utils.constants_tasks import BBH_MC, BBH_COT
from analysis.utils.constants_tasks import PERTURB_COT_TASKS

MODEL_LIST_ALL = []
MODEL_LIST_ALL += MODEL_LADDER_LIST + MODEL_LIST_INTERMEDIATE + MODEL_LIST_MIXES
MODEL_LIST_ALL += OE_EVAL_BASE_MODELS
# MODEL_LIST_ALL += OE_EVAL_INSTRUCT_MODELS
# MODEL_LIST_ALL += MODEL_LIST_FINAL_SIX_CKPTS

TASK_LIST_ALL = []

# TASK_LIST_ALL += RC_TASKS_OLMES
# TASK_LIST_ALL += PARA_TASKS_OLMES
# TASK_LIST_ALL += ENLARGE_TASKS_OLMES
# TASK_LIST_ALL += DISTRACTORS_TASKS_OLMES

# TASK_LIST_ALL += MC_TASKS_COPY_COLORS
# TASK_LIST_ALL += GEN_TASKS_OLMES
# TASK_LIST_ALL += AGI_EVAL_MC + BBH_MC + MMLU_PRO_MC # + MINERVA_MC
TASK_LIST_ALL += AGI_EVAL_COT # + MMLU_PRO_COT

TASK_LIST_ALL += MMLU_PRO_RC
# TASK_LIST_ALL += GEN_TASKS_OLMES_PERTURB_RC
# TASK_LIST_ALL += PERTURB_COT_TASKS


# FOR MODEL LADDER
# TASK_LIST_ALL += [
#     # GSM CoT
#     "gsm8k::olmes",
# 
#     # Minerva CoT (olmes version)
#     "minerva_math_algebra::olmes:full",
#     "minerva_math_counting_and_probability::olmes:full",
#     "minerva_math_geometry::olmes:full",
#     "minerva_math_intermediate_algebra::olmes:full",
#     "minerva_math_number_theory::olmes:full",
#     "minerva_math_prealgebra::olmes:full",
#     "minerva_math_precalculus::olmes:full",
# 
#     # Coding
#     "codex_humaneval:temp0.8",
#     "mbpp::ladder",
#     "mbppplus::ladder",
#     "codex_humanevalplus::ladder",
# ]


# # FOR TESTING
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' not in task] # exclude MMLU (long arg lists may crash beaker! https://github.com/allenai/beaker/issues/5530)
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' in task] # <- MMLU only
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' in task and ':para' in task] # <- MMLU:para only
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'coqa:' not in task] # <- coqa is not setup properly
# MODEL_LIST_ALL = [MODEL_LIST_INTERMEDIATE[1]] # <- only use first model!
# MODEL_LIST_ALL = [OE_EVAL_INSTRUCT_MODELS[0]]
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if 'peteish7' in model or 'peteish13' in model] # <- only peteish7!
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if '-3B-' in model] # <- 3B models!
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if 'qwen2.5-72b' in model or 'qwen2.5-32b' in model or 'qwen2.5-14b' in model] # <- only 3B ladder models!
# TASK_LIST_ALL = SYNTHETIC_TASKS # <- only use synthetic tasks!
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'arc_' not in task]


def run_eval(model_list, task_list, model_type='hf', gpus=1):
    if isinstance(task_list, list): 
        task_list = ' '.join([f'"{task}"' for task in task_list])

    command = f"""
    oe-eval \
        --model {model_list} \
        --task {task_list} \
        --cluster {WEKA_CLUSTERS} \
        --model-type {model_type} \
        --gpus {gpus} \
        --beaker-workspace ai2/davidh \
        --beaker-image davidh/oe-eval-metaeval \
        --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
        --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
        --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
        --recompute-metrics \
        --beaker-priority normal \
        --limit 10000
    """
    command = command.replace('  ', '') # remove extra spacing!
    mb = len(command.encode('utf-8')) / (1024 * 1024) # compute size of command

    print(f'Executing command:\n{command}')
    print(f'Estimated size of all argument strings: {(mb * len(MODEL_LIST_ALL)):.2f} MB (4 MB will crash beaker)')
    
    os.system(command)


def main():
    print('🫢😧 Using a 10K instance limit 🤫🫣')
    print(f'Launching {len(MODEL_LIST_ALL)} models on {len(TASK_LIST_ALL)} tasks (10 second sleep to confirm...)')
    # time.sleep(10)

    for model in MODEL_LIST_ALL:
        if model in OE_EVAL_ALL_MODELS:
            # From my testing, looks like anything less than 4 GPUs on 13B+ models (or Gemma 7B+) breaks
            # Also 70B model do not work on neptune (L40s)
            model_type = 'vllm'
            if model in ['gemma-7b', 'gemma2-9b', "gemma2-2b-instruct", "gemma2-9b-instruct", "gemma2-9b-instruct-SimPO", "llama2-13b", "llama3-70b", "llama3.1-70b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b", "llama3.1-70b-instruct", "qwen2.5-14b-instruct"]:
                gpus = 4
            else:
                gpus = 1 # don't need many GPUs for small models
        elif 'peteish13' in model or 'peteish7' in model:
            model_type = 'vllm'
            gpus = 4
        elif model in MODEL_LIST_MIXES or ('-3B-' in model):
            # Our 3B models have a head size of 208. This is not supported by PagedAttention and will throw errors.
            model_type = 'hf'
            gpus = 1
        else:
            # model_type = 'hf'
            model_type = 'vllm'
            gpus = 1

        run_eval(model, TASK_LIST_ALL, model_type, gpus)


if __name__ == '__main__': main()


# Models that broke on 1 GPU HF
# MODEL_LIST_ALL = [
#     "dclm-1b", "dclm-7b", "gemma-7b", "gemma2-9b", "llama2-7b",
#     "llama3-70b", "llama3.1-70b", "olmo-7b-0424", "qwen2.5-32b", "qwen2.5-72b",
#     "llama-7b",
#     "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm",
# ]


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
