import json

from utils.gpt import openai_init
from utils.__init__ import DATA_DIR, ROOT_DIR

from paraphrase import paraphrase_task, paraphrase_few_shot
from enlarge import enlarge_task, enlarge_task_few_shot
from distractors import add_distractors_task, add_distractors_task_few_shot

from oe_eval.tasks.base_task import Task
from oe_eval.tasks.fewshot_sources import FEWSHOT_SOURCES
from oe_eval.tasks.oe_eval_tasks.paraphrased import PARAPHRASED_TASK_REGISTRY
from oe_eval.tasks.oe_eval_tasks.synthetic import SYNTHETIC_TASK_REGISTRY
from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
from oe_eval.configs.tasks import TASK_CONFIGS

import datasets

ENLARGE_SIZE = 10_000
LIMIT = None


def sanity_check_length(task: Task):
    """ check if there are the same number of generated and original choices """
    task.download()
    docs = task.get_eval_docs(limit=1)

    for doc in docs:
        try:
            assert len(doc['choices']) == len(doc['paraphrased_choices']), f"Lengths do not match: {len(doc['choices'])} != {len(doc['paraphrased_choices'])}"
        except AssertionError as e:
            raise RuntimeError(f'{doc["idx"]}: {e}')


def _test_dataloader(task_name: str, task_cls: Task, config_name: str):
    print(f'Testing dataloader for "{task_name}" on config "{config_name}"')

    olmes_config = TASK_CONFIGS[config_name]
    task: Task = task_cls(
        task_name = olmes_config['task_name'], 
        task_config = olmes_config,
        download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    )

    if ':mc' in task_name: assert 'MC' in task.__class__.__name__, f'Task name {task_name} but class name is {task.__class__.__name__}. Make sure the task_cls was set properly.'

    task.download()
    # sanity_check_length(task)

    try:
        docs = task.get_eval_docs(limit=1)

        # print(json.dumps(docs, indent=4))

        # For some reason, this is not set properly in task __init__
        for field in ['metric_kwargs', 'context_kwargs', 'generation_kwargs']:
            if task.task_config[field] is None:
                task.task_config[field] = {}

        task.build_all_requests()
        task.build_all_requests() # need to test just in case. this is called multiple times if different variations are used, and occasionally breaks the dataloader

        task_instances = task._instances
        example_instance = task_instances[0]

        # print(example_instance)

        # Color the data examples
        if ':para' in task_name:
            c1, c2 = str(34), str(35)
        elif ':rc' in task_name:
            c1, c2 = str(36), str(31)
        elif ':mc' in task_name:
            c1, c2 = str(32), str(31)
        else:
            c1, c2 = str(34), str(35)

        if example_instance.request_type == 'loglikelihood':
            print(f'EXAMPLE REQUEST ({task_name}):\n\033[{c1}m{example_instance.request.context}\033[0m\033[{c2}m{example_instance.request.continuation}\033[0m')
        elif example_instance.request_type == 'generate_until':
            print(f'EXAMPLE REQUEST ({task_name}):\n\033[{c1}m{example_instance.request.context}\033[0m\033[{c2}m{example_instance.label}\033[0m')
            
    except Exception as e:
        raise RuntimeError(f"{task_name} failed on error: {e}")


def test_dataloader():
    """ Test whether the new datasets can be loaded in oe-eval """
    # for task_name, task_cls in PARAPHRASED_TASK_REGISTRY.items():
    for task_name, task_cls in SYNTHETIC_TASK_REGISTRY.items():
        TASKS_TO_INCLUDE = ['arc_easy', 'arc_challenge'] # mmlu, mmlu_computer_security, 'boolq', 'openbookqa', 'winogrande'
        if not any(task in task_name for task in TASKS_TO_INCLUDE): continue

        task_root = task_name.split(':')[0]

        # Find RC task equivalent
        rc_task_name = task_root
        rc_task_cls = TASK_REGISTRY[rc_task_name]
        rc_task_name += ':rc'
        
        # Find MC task equivalent
        mc_task_name = f'{task_root}:mc'
        mc_task_cls = TASK_REGISTRY[mc_task_name]
        mc_config_name = f'{mc_task_name}::olmes'

        _test_dataloader(mc_task_name, mc_task_cls, mc_config_name)

        for version in ['']: # , ':show_options', ':show_options:full_io'
            rc_config_name = f'{rc_task_name}{version}::olmes'
            _test_dataloader(rc_task_name, rc_task_cls, rc_config_name)

        for version in ['prefix_para']: # 'prefix_para', 'suffix_para', 'full_io'
            # Get Para task name
            config_name = f'{task_name}:{version}::olmes'
            _test_dataloader(task_name, task_cls, config_name)


def main(transform):
    openai_init()

    NEW_FEWSHOT_SOURCES = {}

    if transform == 'para':
        task_registry = PARAPHRASED_TASK_REGISTRY
    elif transform == 'enlarge':
        task_registry = TASK_REGISTRY
    elif transform == 'distractors':
        task_registry = TASK_REGISTRY
    else:
        raise ValueError(transform)

    for task_name, task_cls in task_registry.items():
        TASKS_TO_INCLUDE = ['arc_challenge', 'arc_easy'] # mmlu, mmlu_computer_security
        if not any(task in task_name for task in TASKS_TO_INCLUDE): continue

        if transform == 'para':
            # We use the abstract parent class of the paraphrased task class
            task_cls = task_cls.__base__
            olmes_config = TASK_CONFIGS[f'{task_name}::olmes']
        elif transform == 'enlarge' or transform == 'distractors':
            if ':' in task_name: continue # only get the root RC task
            olmes_config = TASK_CONFIGS[f'{task_name}:rc::olmes']
        else:
            ValueError(transform)

        print(f'Processing {task_name}')

        # Manual override for ARC (since there are two ARC classes to inherit)
        if 'arc' in task_name and 'challenge' in task_name:
            olmes_config['dataset_name'] = 'ARC-Challenge'
        elif 'arc' in task_name and 'easy' in task_name:
            olmes_config['dataset_name'] = 'ARC-Easy'

        task: Task = task_cls(
            task_name = olmes_config['task_name'], 
            task_config = olmes_config,
            download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
        )

        if transform == 'para':
            paraphrase_task(task, limit=LIMIT)
            few_shot_f = paraphrase_few_shot
        elif transform == 'enlarge':
            enlarge_task(task, limit=ENLARGE_SIZE)
            few_shot_f = enlarge_task_few_shot
        elif transform == 'distractors':
            add_distractors_task(task, n_new_distractors=4, limit=LIMIT)
            few_shot_f = add_distractors_task_few_shot
        else:
            raise ValueError(transform)

        if 'mmlu' in task_name:
            # MMLU few shot examples are saved as the dev set
            examples = task.fewshot_examples(1000000, None, None)
            docs = few_shot_f(task, examples)
            filepath = f'{DATA_DIR}/{transform}/{task_name}/dev.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(docs, f, ensure_ascii=False, indent=4)
        else:
            orig_few_shot_name = olmes_config['fewshot_source'].replace(f':{transform.capitalize()}', '')
            examples = FEWSHOT_SOURCES[orig_few_shot_name]
            NEW_FEWSHOT_SOURCES[f'{orig_few_shot_name}:{transform.capitalize()}'] = few_shot_f(task, examples)

    # save few shot sources
    # output_file = f'{DATA_DIR}/{transform}/fewshot_sources.json'
    output_file = f'{ROOT_DIR}/oe-eval/oe-eval-internal/oe_eval/tasks/oe_eval_tasks/synthetic/fewshot_sources_{transform}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(NEW_FEWSHOT_SOURCES, f, indent=4, ensure_ascii=False)


if __name__ == '__main__': 
    # main(transform='para')
    # main(transform='enlarge')
    # main(transform='distractors')
    test_dataloader()

    # task_name = 'triviaqa'
    # task_cls = TASK_REGISTRY[task_name]
    # _test_dataloader(task_name, task_cls, config_name='triviaqa::none')
