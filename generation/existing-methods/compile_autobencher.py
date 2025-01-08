import json
import copy
import re
import sys, os
import pandas as pd

sys.path.append('../../') # add this download utility to sys path
sys.path.append('../../generation') # add this download utility to sys path

from analysis.download.hf import push_parquet_to_hf
from generation.methods.distractors import _run_add_distractors_task
from generation.generate import openai_init

def generate_distractors(all_questions):
    # TODO: Generate distractors here (create two cols of 4_distractors and 10_distractors, with 4_distractors_gold_idx and 10_distractors_gold_idx)

    all_questions_tmp = copy.deepcopy(all_questions)

    # Convert to oe-eval format to add distractors
    for entry in all_questions_tmp:
        gold_answer = entry['gold_answer'] if 'gold_answer' in entry else entry['answer']
        entry['query'] = entry['question']
        entry['choices'] = [gold_answer]
        entry['gold'] = 0
    
    openai_init()
    questions_with_distractors = _run_add_distractors_task(n_new_distractors=4, docs=all_questions_tmp)

    assert len(all_questions) == len(questions_with_distractors), 'Not all questions successfully generated distractors!'

    # Paste back the gold and choices indices
    for entry, distractors in zip(all_questions, questions_with_distractors):
        entry['choices'] = distractors['choices']
        entry['gold_idx'] = distractors['gold']

    return all_questions

def merge_datasets(path, pattern, filename):
    # Merge all generated question files
    # question_files = sorted(glob.glob(glob_path))

    question_files = sorted([os.path.join(path, f) for f in os.listdir(path) if re.match(pattern, f)])

    all_questions = []
    for file in question_files:
        print(f'Loading {file}...')
        with open(file, 'r') as f:
            questions = json.load(f)
            
            # Get root category name
            result = re.search(r'/([^/]+?)\.', file)
            root_category = result.group(1) if result and result.group(1) != "" else None
            if root_category.startswith('.'): root_category = None
            for entry in questions:
                entry['root_category'] = root_category

            all_questions.extend(questions)

    # all_questions = all_questions[:30_000]
    # all_questions = generate_distractors(all_questions)

    assert len(all_questions) != 0, all_questions

    with open(f"AutoBencher/data/{filename}.json", "w") as f:
        json.dump(all_questions, f, indent=2)

    # Convert to parquet and push to hf
    df = pd.DataFrame(all_questions)
    parquet_path = f"AutoBencher/data/{filename}.parquet"
    df.to_parquet(parquet_path)

    return parquet_path


def main():
    # Push knowledge QA
    path = 'AutoBencher/KI/'
    pattern = r'.*\.\d+\.KI_questions.json'
    filename = 'combined_ki_questions'

    parquet_path = merge_datasets(path, pattern, filename)

    push_parquet_to_hf(
        parquet_file_path=parquet_path,
        hf_dataset_name="allenai/autobencher-knowledge-qa",
        private=True,
        overwrite=True
    )

    # Push math
    path = 'AutoBencher/MATH/'
    pattern = r'.\d+.*.questions_final.json'
    filename = 'combined_math_questions'

    parquet_path = merge_datasets(path, pattern, filename)

    push_parquet_to_hf(
        parquet_file_path=parquet_path,
        hf_dataset_name="allenai/autobencher-math",
        private=True,
        overwrite=True
    )


if __name__ == '__main__': main()
