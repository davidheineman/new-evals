import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils import weka_to_gcs

from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_MIXES_FINAL, MODEL_LIST_MIXES_FINAL_EXTENDED, MODEL_LIST_INTERMEDIATE, MODEL_LIST_INTERMEDIATE_13B, MODEL_LIST_MIXES, OE_EVAL_BASE_MODELS, OE_EVAL_INSTRUCT_MODELS, OE_EVAL_ALL_MODELS, OE_EVAL_BASE_MODELS_EXTENDED
from analysis.utils.constants_final_6_ckpts import MODEL_LIST_FINAL_SIX_CKPTS
from analysis.utils.constants_models import WEKA_CLUSTERS, GCP_CLUSTERS
from analysis.utils.constants_tasks import MC_TASKS_COPY_COLORS, MISSING_EVALS

# OLMES Classic Tasks
from analysis.utils.constants_tasks import RC_TASKS_OLMES, MC_TASKS_OLMES, PARA_TASKS_OLMES, ENLARGE_TASKS_OLMES, DISTRACTORS_TASKS_OLMES

# OLMES Gen Tasks
from analysis.utils.constants_tasks import GEN_TASKS_OLMES, GEN_TASKS_OLMES_PERTURB_RC

# CoT tasks (mainly Tulu tasks)
from analysis.utils.constants_tasks import AGI_EVAL_MC, AGI_EVAL_RC, AGI_EVAL_COT
from analysis.utils.constants_tasks import MMLU_PRO_MC, MMLU_PRO_RC, MMLU_PRO_COT
from analysis.utils.constants_tasks import MINERVA_MC, MINERVA_COT
from analysis.utils.constants_tasks import BBH_MC, BBH_COT
from analysis.utils.constants_tasks import PERTURB_COT_TASKS

# Perplexity tasks
from analysis.utils.constants_tasks import PALOMA, LLM_COMPRESSION, CUSTOM_LOSS

MODEL_LIST_ALL = []
MODEL_LIST_ALL += MODEL_LADDER_LIST
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE
MODEL_LIST_ALL += OE_EVAL_BASE_MODELS
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_13B # 13B intermediate ckpts
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL # ian's new mixes
MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL_EXTENDED # extended set of data mixes
MODEL_LIST_ALL += OE_EVAL_BASE_MODELS_EXTENDED # OLL 2 leaderboard models

# MODEL_LIST_ALL += MODEL_LIST_MIXES # not used for now
# MODEL_LIST_ALL += OE_EVAL_INSTRUCT_MODELS # not used for now
# MODEL_LIST_ALL += MODEL_LIST_FINAL_SIX_CKPTS # not used for now

TASK_LIST_ALL = []

TASK_LIST_ALL += RC_TASKS_OLMES
TASK_LIST_ALL += PARA_TASKS_OLMES 
TASK_LIST_ALL += ENLARGE_TASKS_OLMES
TASK_LIST_ALL += DISTRACTORS_TASKS_OLMES
TASK_LIST_ALL += MC_TASKS_OLMES

TASK_LIST_ALL += MC_TASKS_COPY_COLORS
TASK_LIST_ALL += GEN_TASKS_OLMES
TASK_LIST_ALL += AGI_EVAL_MC + MMLU_PRO_MC # + MINERVA_MC
TASK_LIST_ALL += AGI_EVAL_COT # + MMLU_PRO_COT
TASK_LIST_ALL += BBH_MC # BPB verison of BBH
TASK_LIST_ALL += BBH_COT

TASK_LIST_ALL += MMLU_PRO_RC + AGI_EVAL_RC
TASK_LIST_ALL += GEN_TASKS_OLMES_PERTURB_RC
TASK_LIST_ALL += PERTURB_COT_TASKS

TASK_LIST_ALL += ['autobencher::none', 'autobencher:mc::none']

TASK_LIST_ALL += [
    # GSM CoT
    "gsm8k::olmes:full",
    # Minerva CoT (olmes version)
    "minerva_math_algebra::olmes:full",
    "minerva_math_counting_and_probability::olmes:full",
    "minerva_math_geometry::olmes:full",
    "minerva_math_intermediate_algebra::olmes:full",
    "minerva_math_number_theory::olmes:full",
    "minerva_math_prealgebra::olmes:full",
    "minerva_math_precalculus::olmes:full",
    # Coding
    "mbpp::ladder",
    "mbppplus::ladder",
    "codex_humaneval:temp0.8",
    "codex_humanevalplus::ladder", 
]

TASK_LIST_ALL += [
    'deepmind_math_large::none',
    'medmcqa:rc::none',
    'medmcqa:mc::none',
    'gsm_plus::none',
    'gsm_symbolic::none',
    'gsm_symbolic_p1::none',
    'gsm_symbolic_p2::none',
    'gpqa::none',
    'minerva_math_500::none', 
]

TASK_LIST_ALL += [
    'aime::none',
]

TASK_LIST_ALL += PALOMA
TASK_LIST_ALL += LLM_COMPRESSION
TASK_LIST_ALL += CUSTOM_LOSS

# # FOR TESTING
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' not in task] # exclude MMLU (long arg lists may crash beaker! https://github.com/allenai/beaker/issues/5530)
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' in task] # <- MMLU only
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' in task and ':para' in task] # <- MMLU:para only
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'coqa:' not in task] # <- coqa is not setup properly
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'hellaswag' in task] # <- hs is not setup properly
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'gsm8k' in task] # <- gsm is not setup properly
# MODEL_LIST_ALL = [MODEL_LIST_INTERMEDIATE[1]] # <- only use first model!
# MODEL_LIST_ALL = [OE_EVAL_INSTRUCT_MODELS[0]]
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if 'peteish7' in model] # <- 3B models!
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if '-3B-5xC' in model or 'peteish13' in model] # <- only peteish13!
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if '-3B-' in model] # <- 3B models!
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if '-1B-10xC' in model or '-3B-10xC' in model] # <- only 1B-10xC !
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if 'falcon_and_cc-1B-5xC' in model or '-3B-10xC' in model] # <- only 1B-10xC !
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if 'qwen2.5-72b' in model or 'qwen2.5-32b' in model or 'qwen2.5-14b' in model] # <- only 3B ladder models!
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if 'qwen2.5-7b' in model]
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if model in ['llama3.1-8b', 'pythia-1b', 'olmoe-1b-7b-0924', 'llama3.2-1b', 'qwen2.5-72b', 'llama2-7b', 'llama3-8b']]
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if model in ['gemma-2b', 'llama2-13b', 'phi-1.5', 'olmo-7b-0724', 'olmo-1b', "llama3.1-70b"]]
# MODEL_LIST_ALL = ["llama3.1-70b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-72b"] # autobencher mc
# MODEL_LIST_ALL = ["qwen2.5-32b", "qwen2.5-14b", "qwen2.5-72b"] # llm compression
# MODEL_LIST_ALL = OE_EVAL_BASE_MODELS
# TASK_LIST_ALL = SYNTHETIC_TASKS # <- only use synthetic tasks!
# TASK_LIST_ALL = AGI_EVAL_MC + BBH_COT
# TASK_LIST_ALL = ['coqa::olmes:full', 'copycolors_4way:mc::none']
# TASK_LIST_ALL = ['gsm_plus::none']
# TASK_LIST_ALL = ['autobencher:mc::none']
# TASK_LIST_ALL = LLM_COMPRESSION
# TASK_LIST_ALL = ['sky_t1::custom_loss', 'numia_math::custom_loss', 'tulu_if::custom_loss']
# TASK_LIST_ALL = PALOMA
# TASK_LIST_ALL = ['arc_challenge:rc::olmes:full']
# MODEL_LIST_ALL = [
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_reddit-4M-5xC/step5735-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/no_flan-4M-5xC/step5735-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-75p-DCLM-baseline-25p-4M-5xC/step5735-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-50p-DCLM-baseline-50p-4M-5xC/step5735-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-90M-5xC/step29901-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-60M-5xC/step29052-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dolma17-25p-DCLM-baseline-75p-4M-5xC/step5735-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/dclm_ft7percentile_fw2-60M-5xC/step29042-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/DCLM-baseline-20M-5xC/step14594-unsharded-hf",
# ]
# MODEL_LIST_ALL = ['weka://oe-training-default/ai2-llm/checkpoints/OLMo-ladder/peteish-moreeval-rerun-1B-1xC/step16279-unsharded-hf'] # model used for question text


def run_eval(model_list, task_list, model_type='hf', gpus=1, limit=None, batch_size=None, save_requests=True):
    if isinstance(task_list, list): 
        task_list = ' '.join([f'"{task}"' for task in task_list])
    if not isinstance(model_list, list): 
        model_list = [model_list]

    cluster_list = GCP_CLUSTERS
    if any('weka://' in model for model in model_list):
        model_list = [weka_to_gcs(model) for model in model_list]

    if len(model_list) == 1: # convert back list -> str
        model_list = model_list[0]

    command = f"""
    oe-eval \
        --model {model_list} \
        --task {task_list} \
        --cluster {cluster_list} \
        --model-type {model_type} \
        --gpus {gpus} \
        --beaker-workspace ai2/ladder-evals \
        --beaker-image davidh/oe-eval-metaeval \
        --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
        --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
        --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
        --recompute-metrics \
        --beaker-priority normal
    """

    # command = f"""
    # oe-eval \
    #     --model {model_list} \
    #     --task {task_list} \
    #     --cluster {cluster_list} \
    #     --model-type {model_type} \
    #     --gpus {gpus} \
    #     --beaker-workspace ai2/lm-eval \
    #     --beaker-image davidh/oe-eval-metaeval \
    #     --gantry-secret-aws-access-key-id lucas_AWS_ACCESS_KEY_ID \
    #     --gantry-secret-aws-secret-access lucas_AWS_SECRET_ACCESS_KEY \
    #     --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
    #     --recompute-metrics \
    #     --beaker-priority high
    # """
    # --run-local \

    command = command.replace('\n', '').replace('  ', '') # remove extra spacing!
    if limit is not None: 
        print(f'🫢😧 Using a {limit} instance limit 🤫🫣')
        command += f" --limit {limit}"
    if batch_size is not None: 
        print(f'Using a batch_size of {batch_size}')
        command += f" --batch-size {batch_size}"
    if not save_requests:
        command += ' --save-raw-requests false --delete-raw-requests'
    mb = len(command.encode('utf-8')) / (1024 * 1024) # compute size of command

    print(f'Executing command:\n{command}')
    print(f'Estimated size of all argument strings: {(mb * len(MODEL_LIST_ALL)):.2f} MB (4 MB will crash beaker)')
    
    os.system(command)


def main():
    print(f'Launching {len(MODEL_LIST_ALL)} models on {len(TASK_LIST_ALL)} tasks (5 second sleep to confirm...)')
    # time.sleep(5)

    # for (model, missing_tasks) in MISSING_EVALS:
    #     task_list = []
    #     for missing_task in missing_tasks:
    #         task_list += [task_name for task_name in TASK_LIST_ALL if missing_task in task_name]

    #     if len(task_list) == 0:
    #         # print(f'Cant find tasks for {missing_task}')
    #         continue
    #         # raise

    task_list = TASK_LIST_ALL
    for model in MODEL_LIST_ALL:

        batch_size = None
        save_requests = True

        # batch_size = 4 # TMP OVERRIDE FOR LADDER MODELS

        if model in OE_EVAL_ALL_MODELS:
            # From my testing, looks like anything less than 4 GPUs on 13B+ models (or Gemma 7B+) breaks
            # Also 70B model do not work on neptune (L40s)
            model_type = 'vllm'
            if model in ['gemma-7b', 'gemma2-9b', "gemma2-2b-instruct", "gemma2-9b-instruct", "gemma2-9b-instruct-SimPO", "llama2-13b", "llama3-70b", "llama3.1-70b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b", "llama3.1-70b-instruct", "qwen2.5-14b-instruct"] or '32B' in model or '72B' in model or '22B' in model or '15b' in model or '40b' in model or '110B' in model or '70B' in model:
                gpus = 4
            else:
                gpus = 1 # don't need many GPUs for small models
        elif 'peteish13' in model or 'peteish7' in model:
            model_type = 'vllm'
            gpus = 4
        elif model in MODEL_LIST_MIXES + MODEL_LIST_MIXES_FINAL + MODEL_LIST_MIXES_FINAL_EXTENDED or ('-3B-' in model) or model in [weka_to_gcs(m) for m in MODEL_LIST_MIXES + MODEL_LIST_MIXES_FINAL + MODEL_LIST_MIXES_FINAL_EXTENDED]:
            # Our 3B models have a head size of 208. This is not supported by PagedAttention and will throw errors.
            model_type = 'hf'
            gpus = 1
        else:
            # model_type = 'hf'
            model_type = 'vllm'
            gpus = 1

        if any(task in PALOMA + LLM_COMPRESSION + CUSTOM_LOSS for task in task_list):
            save_requests = False # don't save the perplexity files
            model_type = 'hf' # we can only run perplexity on hf for now
            if model in OE_EVAL_BASE_MODELS or '10xC' in model:
                batch_size = 1 # larger corpora will silent fail

        run_eval(
            model_list=model, 
            task_list=task_list, 
            model_type=model_type, 
            gpus=gpus,
            # limit=10_000,
            batch_size=batch_size,
            save_requests=save_requests
        )


if __name__ == '__main__': main()


# Models that broke on 1 GPU HF
# MODEL_LIST_ALL = [
#     "dclm-1b", "dclm-7b", "gemma-7b", "gemma2-9b", "llama2-7b",
#     "llama3-70b", "llama3.1-70b", "olmo-7b-0424", "qwen2.5-32b", "qwen2.5-72b",
#     "llama-7b",
#     "weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/step476848-hf-vllm",
# ]

# Scaling data constrained LLMs (https://arxiv.org/abs/2305.16264)
# https://huggingface.co/datablations#models / https://github.com/huggingface/datablations

# (probably your best bet) 20M to 3.3B:
# https://huggingface.co/KempnerInstituteAI/loss-to-loss
