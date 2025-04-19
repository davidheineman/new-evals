import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')

# Load observational model sizes data
with open(os.path.join(ROOT_DIR, 'analysis/utils/model_sizes.json')) as f:
    MODEL_SIZES = json.load(f)

def get_title_from_task(task):
    if isinstance(task, list):
        assert len(task) > 0, f'Seeing empty array passed as a task: {task}'
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
    task = get_title_from_task(task)
    mapping = {
        'arc_challenge:mc': 'ARC Challenge MC',
        'arc_challenge': 'ARC Challenge',
        'arc_easy:mc': 'ARC Easy MC', 
        'arc_easy': 'ARC Easy', 
        'autobencher:mc': 'Autobencher MC',
        'autobencher': 'AutoBencher',
        'boolq:mc': 'BoolQ MC',
        'boolq': 'BoolQ',
        'codex_humaneval': 'HumanEval',
        'codex_humanevalplus': 'HumanEval+',
        'csqa:mc': 'CommonsenseQA MC',
        'csqa': 'CommonsenseQA',
        'drop': 'DROP',
        'gsm8k': 'GSM8K',
        'hellaswag:mc': 'HellaSwag MC',
        'hellaswag': 'HellaSwag',
        'jeopardy': 'Jeopardy',
        'mbpp': 'MBPP',
        'mbppplus': 'MBPP+',
        'minerva': 'Minerva MATH',
        'mmlu_mc': 'MMLU MC',
        'mmlu': 'MMLU',
        'olmes_core9_mc': 'OLMES Core 9 MC',
        'olmes_core9': 'OLMES Core 9',
        'olmes_gen': 'OLMES Gen',
        'openbookqa:mc': 'OpenBookQA MC',
        'openbookqa': 'OpenBookQA',
        'paloma_c4_en': 'Paloma C4',
        'paloma_m2d2_s2orc_unsplit': 'Paloma M2D2',
        'piqa:mc': 'PIQA MC',
        'piqa': 'PIQA',
        'socialiqa:mc': 'SocialIQA MC',
        'socialiqa': 'SocialIQA', 
        'squad': 'SQuAD',
        'triviaqa': 'TriviaQA',
        'winogrande:mc': 'WinoGrande MC',
        'winogrande': 'WinoGrande',
        'agi_eval': 'AGI Eval',
        'aime': 'AIME',
        'bbh': 'BBH',
        'gsm_plus': 'GSM+',
        'gsm_symbolic_main': 'GSM Symbolic',
        'gsm_symbolic_p1': 'GSM Symbolic P1',
        'gsm_symbolic_p2': 'GSM Symbolic P2', 
        'medmcqa': 'MedMCQA',
        'minerva_math_500': 'Minerva MATH 500',
        'mmlu_pro': 'MMLU Pro'
    }
    if task not in mapping:
        print(f"Task not in mapping: {task}")
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


# Observational models excluded for low performance / broken evals
EXCLUDED_OBS_MODELS = [
    'pythia',
    'phi-1',
    'olmo-1b-0724-hf',
    'stablelm-base-alpha-7b',
    'gemma-2-27b',
    'gemma-2',
    'gemma-7b'
]

def extract_flops(model_alias):
    if model_alias not in MODEL_SIZES:
        return None, False
    
    is_excluded_observational = any(excluded_alias in model_alias.lower() for excluded_alias in EXCLUDED_OBS_MODELS)
    if is_excluded_observational:
        return None, False
    
    active_params = MODEL_SIZES[model_alias]["active_params_B"] * 1e9  # Convert B to raw number
    tokens = MODEL_SIZES[model_alias]["toks_T"] * 1e12 if MODEL_SIZES[model_alias]["toks_T"] else 0  # Convert T to raw number
    return 6 * active_params * tokens, True  # 6ND calculation and observational status
