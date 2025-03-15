import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')

def get_title_from_task(task):
    if isinstance(task, list):
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
