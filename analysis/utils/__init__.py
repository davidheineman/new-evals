import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')

def get_title_from_task(task):
    if isinstance(task, list):
        if len(task) == 1:
            return task[0]
        title_mapping = {
            'mmlu_pro_': 'mmlu_pro',
            'mmlu_abstract_algebra:mc': 'mmlu_mc',
            'mmlu': 'mmlu',
            'minerva': 'minerva',
            'agi_eval': 'agi_eval',
            'bbh': 'bbh',
            'arc_challenge:para': 'olmes_core9_para',
            'arc_challenge:distractors': 'olmes_core9_distractors',
            'arc_challenge:enlarge': 'olmes_core9_enlarge',
            'arc_challenge:mc': 'olmes_core9_mc',
            'arc_challenge': 'olmes_core9',
            'drop': 'olmes_gen',
        }
        for key, title in title_mapping.items():
            if key in task[0]:
                return title
        return 'aggregate'
    return task

def get_pretty_task_name(task):
    """Map task names to prettier display names"""
    mapping = {
        'arc_challenge': 'ARC Challenge',
        'arc_easy': 'ARC Easy', 
        'autobencher': 'AutoBencher',
        'boolq': 'BoolQ',
        'codex_humaneval': 'HumanEval',
        'codex_humanevalplus': 'HumanEval+',
        'csqa': 'CommonsenseQA',
        'drop': 'DROP',
        'gsm8k': 'GSM8K',
        'hellaswag': 'HellaSwag',
        'jeopardy': 'Jeopardy',
        'mbpp': 'MBPP',
        'mbppplus': 'MBPP+',
        'minerva': 'Minerva',
        'mmlu': 'MMLU',
        'olmes_core9': 'OLMES Core 9',
        'olmes_gen': 'OLMES Gen',
        'openbookqa': 'OpenBookQA',
        'paloma_c4_en': 'Paloma C4',
        'paloma_m2d2_s2orc_unsplit': 'Paloma M2D2',
        'piqa': 'PIQA',
        'socialiqa': 'SocialIQA', 
        'squad': 'SQuAD',
        'triviaqa': 'TriviaQA',
        'winogrande': 'WinoGrande'
    }
    return mapping.get(task, task)

def weka_to_gcs(model_name):
    if 'weka://' in model_name:
        return f"gs://ai2-llm/checkpoints/davidh/{model_name.split('checkpoints/')[1]}"
    else:
        return model_name

def fix_model_path(name):
    if name.endswith('peteish7/step928646-hf'):
        name = 'peteish7/step928646-hf-vllm-2'
    name = name.replace('OLMoE-1B-7B-0924', 'olmoe-1b-7b-0924')
    name = name.replace('OLMo-7B-hf', 'olmo-7b')
    name = name.replace('phi-1.5', 'phi-1_5')
    name = name.replace('Qwen2-1.5B', 'qwen2-1.5b')
    name = name.replace('Qwen2.5-3B', 'qwen2.5-3b')
    name = name.replace('deepseek-7b', 'deepseek-llm-7b-base')
    return name

def extract_size(model_name):
    """ Extract size from model name 'falcon-11B' => 11_000_000_000 """
    parts = model_name.split('-')
    for part in parts:
        if part.endswith('M') or part.endswith('m'):
            try:
                return float(part[:-1]) * 1e6
            except ValueError as e:
                continue
        elif part.endswith('B') or part.endswith('b'):
            try:
                return float(part[:-1]) * 1e9
            except ValueError as e:
                continue
    return None