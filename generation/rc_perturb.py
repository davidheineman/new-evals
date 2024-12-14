from collections import defaultdict
import re
import json
import random
import copy

from pathlib import Path

from utils.gpt import generate_gpt, print_estimate_cost
from utils.parser import _parse_choices
from utils import DATA_DIR

from oe_eval.tasks.base_task import Task

random.seed(42)

SEP = '>>>'
N_RETRIES = 5


LLM_PROMPT = """You are given examples of QUESTIONS, with an ANSWER EXPLANATION and an ANSWER. Please generate {n_choices} new INCORRECT EXPLANATIONS and INCORRECT ANSWERS that are incorrect, but plausible explanation paths to the question. You can take the example EXPLANATION and insert a subtle error so the final output is incorrect. Please add an explanation as to what error was inserted with a [DIFF][/DIFF] tag. Make sure to add your list of NEW INCORRECT ANSWERS at the end!

For example:

{few_shot_examples}


Now you try!

QUESTION: {question}

CORRECT ANSWER EXPLANATION: {cot}

CORRECT ANSWER: {answer}

NEW INCORRECT CHAIN-OF-THOUGHT CHOICES:"""

FEW_SHOT_TEMPLATE = """QUESTION: {example_question}

CORRECT ANSWER EXPLANATION: {example_cot}

CORRECT ANSWER: {example_answer}

NEW INCORRECT CHAIN-OF-THOUGHT CHOICES: {example_incorrect_cots}

NEW INCORRECT ANSWERS: {example_incorrect_answers}"""


# This is the first question of GSM (probably should take the example from elsewhere)
DEFAULT_EXAMPLE_QUESTION = ["Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"]

DEFAULT_EXAMPLE_COT = ["Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"]

DEFAULT_EXAMPLE_ANSWER = [18]

DEFAULT_EXAMPLE_INCORRECT_COTS = [f"""
{SEP} Janet sells 16 - 3 = <<16-3=13>>13 duck eggs a day because she only considers the eggs she eats for breakfast. Since she bakes muffins with the eggs she saves, she sells all the remaining eggs for $2 each, earning 13 * 2 = $<<13*2=26>>26 daily.\n#### 26 [DIFF] Incorrectly ignores the 4 eggs used for muffins in the calculation of the remaining eggs sold.[/DIFF]
{SEP} Janet has 16 eggs in total and after eating 3 for breakfast, she only sells 4 eggs at the market since she uses the rest for baking. Thus, she makes 4 * 2 = $<<4*2=8>>8 every day at the farmer's market.\n#### 8 [DIFF] Miscalculates the number of eggs available for sale by misinterpreting how many are used for baking versus how many are sold.[/DIFF]
{SEP} Janet consumes 3 eggs and uses 4 for baking, which totals 7 eggs consumed. Therefore, she sells the remaining 16 - 7 = <<16-7=9>>9 eggs for $2 each, giving her a total of 9 * 2 = $<<9*2=16>>16 at the farmer’s market.\n#### 16 [DIFF] The calculation is correct but misunderstands the problem; the total eggs used for breakfast and baking should be subtracted from 16, not counted separately.[/DIFF]
{SEP} Janet sells all 16 eggs for $2 each, but after keeping 3 for breakfast and using 4 for muffins, she still has 16 - 3 = <<16-3=13>>13 eggs to sell, making 13 * 2 = $<<13*2=26>>26 every day.\n#### 26 [DIFF] Assumes all 16 eggs can be sold despite consuming some, leading to an inflated total.[/DIFF]
"""]

DEFAULT_EXAMPLE_INCORRECT_ANSWERS = [[26, 8, 16, 26]]



def perturb_rc_task(task: Task, n_cots: int, limit=None):
    task.download()
    docs = task.get_eval_docs(limit=limit)

    docs = _run_perturb_rc_task(task, n_cots, docs, limit=limit)

    filepath = f'{DATA_DIR}/perturb_rc/{task.task_config["task_name"]}:perturb_rc/{task.task_config["split"]}.json'

    # save reformatted benchmark
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def perturb_rc_task_few_shot(task: Task, few_shot_examples: list[dict], n_choices: int = 4): 
    """
    Most tasks already have few-shot chain-of-thought prompts. We'll use those.
    We need these keys: 'choices' 'gold' 'query' 'id'
    """
    few_shot_examples = [task._process_doc(doc) for doc in few_shot_examples]

    # cot_docs = [task._process_doc(doc) for doc in few_shot_examples]
    # few_shot_examples = _run_perturb_rc_task(task, cot_task, n_cots, docs, cot_docs)

    c1, c2 = str(31), str(32)
    print(f'\033[{c1}mExample few shot doc: \033[0m\033[{c2}m{few_shot_examples[0]}\033[0m')

    few_shot_parsed = []
    for i, example_doc in enumerate(few_shot_examples):
        question, answer, gold_cot = parse_doc(example_doc)

        few_shot_parsed += [{
            'choices': [gold_cot],
            'gold': 0,
            'query': question,
            'id': f'example_{i}'
        }]


    return few_shot_parsed


def parse_doc(doc):
    # Get question
    if 'question' in doc:
        question = doc['question']
    elif 'problem' in doc:
        question = doc['problem']
    elif 'query' in doc:
        question = doc['query']
        question = question.replace('Question: ', '').replace("\nAnswer: Let's think step by step.", "")
    elif 'prompt' in doc:
        question = doc['prompt']
    else:
        raise KeyError(doc)
    
    # Get gold answer (verifiable benchmarks, like IF Eval, don't have a gold answer)
    answer = None
    if 'short_answer' in doc:
        answer = doc['short_answer']
    elif 'answer' in doc:
        answer = doc['answer']
    elif 'label' in doc:
        answer = doc['label']
    
    return question, answer


def _run_perturb_rc_task(task: Task, docs: dict, n_choices: int = 4):
    prompts = []

    for i, doc in enumerate(docs):
        if i == 0: 
            c1, c2 = str(31), str(32)
            print(f'\033[{c1}mExample doc: \033[0m\033[{c2}m{doc}\033[0m')

        question, answer = parse_doc(doc)

        # construct few show examples
        few_shot_question, few_shot_cot, few_shot_answers, few_shot_incorrect_cots, few_shot_incorrect_answers = None, None, None, None, None

        if few_shot_question is None:          few_shot_question = DEFAULT_EXAMPLE_QUESTION
        if few_shot_cot is None:               few_shot_cot = DEFAULT_EXAMPLE_COT
        if few_shot_answers is None:           few_shot_answers = DEFAULT_EXAMPLE_ANSWER
        if few_shot_incorrect_cots is None:    few_shot_incorrect_cots = DEFAULT_EXAMPLE_INCORRECT_COTS
        if few_shot_incorrect_answers is None: few_shot_incorrect_answers = DEFAULT_EXAMPLE_INCORRECT_ANSWERS

        few_shot_incorrect_answers = [f'\n{SEP} ' + f'\n{SEP} '.join([str(ic) for ic in fs_ic]) for fs_ic in few_shot_incorrect_answers]

        few_shot_text = '\n\n'.join([
            FEW_SHOT_TEMPLATE.format(
                example_question=fs_q,
                example_cot=fs_c,
                example_answer=fs_a,
                example_incorrect_cots=fs_ic.rstrip(),
                example_incorrect_answers=fs_ia
            ) for fs_q, fs_c, fs_a, fs_ic, fs_ia in zip(few_shot_question, few_shot_cot, few_shot_answers, few_shot_incorrect_cots, few_shot_incorrect_answers)
        ])

        # construct GPT prompt
        prompt = LLM_PROMPT.format(
            n_choices=n_choices,
            few_shot_examples=few_shot_text,
            question=question,
            cot=gold_cot,
            answer=answer,
        )

        if i == 0: print("\033[94m" + prompt + "\033[0m")

        prompts += [prompt]

    print_estimate_cost(prompts, model='gpt-4o-mini', input_cost=0.15, output_cost=0.6)
    # print_estimate_cost(prompts, model='gpt-4o', input_cost=2.5, output_cost=10)

    responses = generate_gpt(prompts, model='gpt-4o-mini', max_tokens=4096)
    
    # Attempt parsing responses, with retries on failure
    for i, (prompt, response, doc) in enumerate(zip(prompts, responses, docs)):
        n_retries = 0
        while n_retries < N_RETRIES:
            try:
                if isinstance(response, list) and len(response) == 1: response = response[0] # collapse a list of a single string to a string

                response = response.replace('NEW INCORRECT CHAIN-OF-THOUGHT CHOICES:', '')
                response = response.lstrip()

                # Separate the two sets of responses from the model
                response_incorrect_cots, response_incorrect_answers = response.split('NEW INCORRECT ANSWERS:')
                response_incorrect_cots = response_incorrect_cots.strip()
                response_incorrect_answers = response_incorrect_answers.strip()

                # Parse answer choices (we can get away with n-1 answers)
                response_choices = _parse_choices(response_incorrect_cots, n_choices=[n_choices, n_choices-1], sep=SEP)
                response_answers = _parse_choices(response_incorrect_answers, n_choices=[n_choices, n_choices-1], sep=SEP)

                # TODO: VERIFY ANSWERS ARE INCORRECT

                # remove reason text
                def remove_reason_tags(text):
                    return re.sub(r'\[DIFF\].*?\[/DIFF\]', '', text, flags=re.DOTALL).rstrip()
                response_choices = [remove_reason_tags(r) for r in response_choices]
            except (IndexError, AttributeError, AssertionError, ValueError, TypeError) as e:
                # raise RuntimeError(repr(response))
                print(f"Error parsing response: {e}\nResponse:\n{repr(response)}")
                response_choices = None

            if response_choices is None:
                # Parsing failed, attempt to retry
                c1 = str(31)
                print(f'\033[{c1}mParsing failed, retrying... ({n_retries}/{N_RETRIES})\033[0m')
                response = generate_gpt([prompt], model='gpt-4o-mini', max_tokens=4096)
            else:
                break
            n_retries += 1

        if response_choices is not None:
            # TODO: verify incorrect cots are incorrect

            # if 'id' in doc:
            #     id = doc['id']
            # elif 'index' in doc:
            #     id = doc['index']
            # elif 'key' in doc:
            #     id = doc['key']
            # else:
            #     raise KeyError(doc)
            id = i

            gold_cot = doc['gold_cot']
            query = task.doc_to_text(doc)

            # shuffle perturbed cot
            choices = [gold_cot] + response_choices[:n_choices]
            random.shuffle(choices)

            # construct final query
            docs[i]['choices'] = choices
            docs[i]['gold'] = choices.index(gold_cot)
            docs[i]['query'] = query
            docs[i]['id'] = f'cot_perturb_{id}'

    return docs
