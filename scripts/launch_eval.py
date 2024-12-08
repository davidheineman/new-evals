from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_INTERMEDIATE, MODEL_LIST_MIXES, OE_EVAL_OFFICIAL_MODELS
from analysis.utils.constants_models import RC_TASKS_OLMES, PARA_TASKS_OLMES
from analysis.utils.constants_models import WEKA_CLUSTERS

import os, time

MODEL_LIST_ALL = MODEL_LADDER_LIST + MODEL_LIST_INTERMEDIATE + MODEL_LIST_MIXES
TASK_LIST_ALL = RC_TASKS_OLMES + PARA_TASKS_OLMES

SYNTHETIC_TASKS = [
    "arc_easy:enlarge::olmes:full",
    "arc_challenge:enlarge::olmes:full",
    "arc_easy:distractors::olmes:full",
    "arc_challenge:distractors::olmes:full",
]
TASK_LIST_ALL += SYNTHETIC_TASKS

# # FOR TESTING
# MODEL_LIST_ALL = [MODEL_LIST_ALL[0]] # <- only use first model!
# TASK_LIST_ALL = SYNTHETIC_TASKS # <- only use synthetic tasks!
MODEL_LIST_ALL = OE_EVAL_OFFICIAL_MODELS
# TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'mmlu_' not in task]
TASK_LIST_ALL = [task for task in TASK_LIST_ALL if 'arc' in task]

def run_eval(model_list, task_list):
    if isinstance(task_list, list): 
        task_list = ' '.join(task_list)

    command = f"""
    oe-eval \
        --model {model_list} \
        --task {task_list} \
        --cluster {WEKA_CLUSTERS} \
        --model-type hf \
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